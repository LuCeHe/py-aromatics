import random, string

import torch

forward_normalizer = {}
forward_centers = {}
backward_normalizer = {}
backward_centers = {}


class SurrogateGradNormalizable(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, id, sg_curve, input_f, input_grad_f, sgout_f):
        ctx.save_for_backward(input_)
        ctx.id = id
        ctx.sg_curve = sg_curve
        ctx.input_f = input_f
        ctx.input_grad_f = input_grad_f
        ctx.sgout_f = sgout_f

        out = (input_ > 0).type(input_.dtype)
        # print('ehrer', torch.mean(out).cpu().detach().numpy())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()

        grad_input = ctx.input_grad_f(grad_input, ctx.id)
        input_ = ctx.input_f(input_, ctx.id)

        sgout = ctx.sg_curve(input_)

        sgout = ctx.sgout_f(sgout, ctx.id)
        grad = grad_input * sgout

        return grad, None, None, None, None, None


class ConditionedSG(torch.nn.Module):
    def __init__(self, rule, on_ingrad=False, forwback=False, curve_name='dfastsigmoid', continuous=False,
                 normalized_curve=False, sgoutn=False):
        super().__init__()

        global forward_normalizer, forward_centers, backward_normalizer, backward_centers
        self.act = SurrogateGradNormalizable.apply

        input_normalizer = lambda input_, id: input_
        ingrad_normalizer = lambda input_, id: input_
        sgout_normalizer = lambda input_, id: input_

        if rule == 'IV':

            def f(centers, normalizer):
                def _f(input_, id):
                    if not id in normalizer.keys() or continuous:
                        normalizer[id] = torch.std(input_)
                    return input_ / normalizer[id]
                return _f

        elif rule == 'I':
            def f(centers, normalizer):
                def _f(input_, id):
                    if not id in centers.keys() or continuous:
                        centers[id] = torch.mean(input_)
                    center = centers[id]
                    return input_ - center
                return _f

        elif rule == 'I_IV':
            def f(centers, normalizer):
                def _f(input_, id):
                    if not id in normalizer.keys() or continuous:
                        normalizer[id] = torch.std(input_)
                    std = normalizer[id]

                    if not id in centers.keys() or continuous:
                        centers[id] = torch.mean(input_)
                    center = centers[id]

                    return (input_ - center) / std
                return _f
        elif rule == '0':
            def f(centers, normalizer):
                def _f(input_, id):
                    return input_
                return _f
        else:
            raise Exception('Unknown rule: {}'.format(rule))


        if sgoutn:
            sgout_normalizer = f(centers=forward_centers, normalizer=forward_normalizer)

        elif forwback:
            input_normalizer = f(centers=forward_centers, normalizer=forward_normalizer)
            ingrad_normalizer = f(centers=backward_centers, normalizer=backward_normalizer)

        elif on_ingrad:
            ingrad_normalizer = f(centers=backward_centers, normalizer=backward_normalizer)

        else:
            input_normalizer = f(centers=forward_centers, normalizer=forward_normalizer)

        self.input_normalizer = input_normalizer
        self.ingrad_normalizer = ingrad_normalizer
        self.sgout_normalizer = sgout_normalizer

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
        return self.act(x, self.id, self.sg_curve, self.input_normalizer, self.ingrad_normalizer, self.sgout_normalizer)
