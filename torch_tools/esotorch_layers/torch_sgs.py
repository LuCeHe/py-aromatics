import random, string

import torch

sg_normalizer = {}
sg_centers = {}


class SurrogateGradNormalizable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id, sg_curve, input_normalizer):
        ctx.save_for_backward(input_)
        ctx.id = id
        ctx.sg_curve = sg_curve
        ctx.input_normalizer = input_normalizer

        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        input_ = ctx.input_normalizer(input_, ctx.id)

        grad = grad_input * ctx.sg_curve(input_)

        return grad, None, None, None


class ConditionedSG(torch.nn.Module):
    def __init__(self, rule, curve_name='dfastsigmoid', continuous=False):
        super().__init__()

        self.act = SurrogateGradNormalizable.apply

        if rule == '0':
            input_normalizer = lambda input_, id: input_

        elif rule == 'IV':
            def input_normalizer(input_, id):
                global sg_normalizer
                if not id in sg_normalizer.keys() or continuous:
                    sg_normalizer[id] = torch.std(input_)
                return input_ / sg_normalizer[id]

        elif rule == 'I':
            def input_normalizer(input_, id):
                global sg_centers
                if not id in sg_centers.keys() or continuous:
                    sg_centers[id] = torch.mean(input_)
                center = sg_centers[id]
                return input_ - center

        elif rule == 'I_IV':

            def input_normalizer(input_, id):
                global sg_normalizer
                if not id in sg_normalizer.keys() or continuous:
                    sg_normalizer[id] = torch.std(input_)
                std = sg_normalizer[id]

                global sg_centers
                if not id in sg_centers.keys() or continuous:
                    sg_centers[id] = torch.mean(input_)
                center = sg_centers[id]

                return (input_ - center) / std
        else:
            raise Exception('Unknown rule: {}'.format(rule))

        self.input_normalizer = input_normalizer

        if curve_name == 'dfastsigmoid':
            self.sg_curve = lambda x: 1 / (10 * torch.abs(x) + 1.0) ** 2

        elif curve_name == 'triangular':
            self.sg_curve = lambda x: torch.maximum(1 - 10 * torch.abs(x), torch.zeros_like(x))

        elif curve_name == 'rectangular':
            self.sg_curve = lambda x: torch.abs(10 * x) < 1 / 2
        else:
            raise Exception('Unknown curve: {}'.format(curve_name))

        characters = string.ascii_letters + string.digits
        self.id = ''.join(random.choice(characters) for _ in range(5))

    def forward(self, x):
        return self.act(x, self.id, self.sg_curve, self.input_normalizer)
