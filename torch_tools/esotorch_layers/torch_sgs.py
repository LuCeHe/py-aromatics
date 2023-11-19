import random, string

import torch

sg_normalizer = {}
sg_centers = {}
sg_curve = lambda x: 1 / (10 * torch.abs(x) + 1.0) ** 2


class SurrogateGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id):
        ctx.save_for_backward(input_)
        ctx.id = id
        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global sg_curve
        grad = grad_input * sg_curve(input_)

        return grad, None


class SurrogateGradIV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id):
        ctx.save_for_backward(input_)
        ctx.id = id
        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global sg_normalizer
        if not ctx.id in sg_normalizer.keys():
            sg_normalizer[ctx.id] = torch.std(input_)
        input_ = input_ / sg_normalizer[ctx.id]

        global sg_curve
        grad = grad_input * sg_curve(input_)

        return grad, None


class SurrogateGradI_IV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id):
        ctx.save_for_backward(input_)
        ctx.id = id
        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global sg_centers
        if not ctx.id in sg_centers.keys():
            sg_centers[ctx.id] = torch.mean(grad_input)

        global sg_normalizer
        if not ctx.id in sg_normalizer.keys():
            sg_normalizer[ctx.id] = torch.std(input_)

        center = sg_centers[ctx.id]
        std = sg_normalizer[ctx.id]
        input_ = (input_ - center) / std

        global sg_curve
        grad = grad_input * sg_curve(input_)

        return grad, None


class SurrogateGradI(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id):
        ctx.save_for_backward(input_)
        ctx.id = id
        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        global sg_centers
        if not ctx.id in sg_centers.keys():
            sg_centers[ctx.id] = torch.mean(grad_input)

        center = sg_centers[ctx.id]
        input_ = input_ - center

        global sg_curve
        grad = grad_input * sg_curve(input_)
        return grad, None


class ConditionedSG(torch.nn.Module):
    def __init__(self, rule, curve_name='dfastsigmoid'):
        super().__init__()
        if rule == '0':
            self.act = SurrogateGrad.apply
        elif rule == 'IV':
            self.act = SurrogateGradIV.apply
        elif rule == 'I':
            self.act = SurrogateGradI.apply
        elif rule == 'I_IV':
            self.act = SurrogateGradI_IV.apply
        else:
            raise Exception('Unknown rule: {}'.format(rule))

        if curve_name == 'dfastsigmoid':
            global sg_curve
            sg_curve = lambda x: 1 / (10 * torch.abs(x) + 1.0) ** 2

        elif curve_name == 'triangular':
            global sg_curve
            sg_curve = lambda x: torch.maximum(1 - 10 * torch.abs(x), torch.zeros_like(x))

        elif curve_name == 'rectangular':
            global sg_curve
            sg_curve = lambda x: torch.abs(10 * x) < 1 / 2

        characters = string.ascii_letters + string.digits
        self.id = ''.join(random.choice(characters) for _ in range(5))

    def forward(self, x):
        return self.act(x, self.id)
