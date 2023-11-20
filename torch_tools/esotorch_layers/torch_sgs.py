import random, string

import torch

sg_normalizer = {}
sg_centers = {}


class SurrogateGradNormalizable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id, sg_curve, input_f, input_grad_f):
        ctx.save_for_backward(input_)
        ctx.id = id
        ctx.sg_curve = sg_curve
        ctx.input_f = input_f
        ctx.input_grad_f = input_grad_f

        return (input_ > 0).type(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = ctx.input_grad_f(grad_input, ctx.id)
        input_ = ctx.input_f(input_, ctx.id)

        grad = grad_input * ctx.sg_curve(input_)

        return grad, None, None, None, None


class ConditionedSG(torch.nn.Module):
    def __init__(self, rule, on_ingrad=False, curve_name='dfastsigmoid', continuous=False, normalized_curve=False):
        super().__init__()

        self.act = SurrogateGradNormalizable.apply

        input_normalizer = lambda input_, id: input_
        ingrad_normalizer = lambda input_, id: input_

        if rule == 'IV':
            if not on_ingrad:
                def input_normalizer(input_, id):
                    global sg_normalizer
                    if not id in sg_normalizer.keys() or continuous:
                        sg_normalizer[id] = torch.std(input_)
                    return input_ / sg_normalizer[id]
            else:
                def ingrad_normalizer(input_, id):
                    global sg_normalizer
                    if not id in sg_normalizer.keys() or continuous:
                        sg_normalizer[id] = torch.std(input_)
                    return input_ / sg_normalizer[id]

        elif rule == 'I':
            if not on_ingrad:
                def input_normalizer(input_, id):
                    global sg_centers
                    if not id in sg_centers.keys() or continuous:
                        sg_centers[id] = torch.mean(input_)
                    center = sg_centers[id]
                    return input_ - center
            else:
                def ingrad_normalizer(input_, id):
                    global sg_centers
                    if not id in sg_centers.keys() or continuous:
                        sg_centers[id] = torch.mean(input_)
                    center = sg_centers[id]
                    return input_ - center

        elif rule == 'I_IV':
            if not on_ingrad:
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
                def ingrad_normalizer(input_, id):
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
        self.ingrad_normalizer = ingrad_normalizer

        if curve_name == 'dfastsigmoid':
            m = 10 if not normalized_curve else 2
            self.sg_curve = lambda x: 1 / (m * torch.abs(x) + 1.0) ** 2

        elif curve_name == 'triangular':
            m = 10 if not normalized_curve else 1
            self.sg_curve = lambda x: torch.maximum(1 - m * torch.abs(x), torch.zeros_like(x))

        elif curve_name == 'rectangular':
            m = 10 if not normalized_curve else 1
            self.sg_curve = lambda x: torch.abs(m * x) < 1 / 2
        else:
            raise Exception('Unknown curve: {}'.format(curve_name))

        characters = string.ascii_letters + string.digits
        self.id = ''.join(random.choice(characters) for _ in range(5))

    def forward(self, x):
        return self.act(x, self.id, self.sg_curve, self.input_normalizer, self.ingrad_normalizer)
