import re
from typing import Optional, Union, Callable, List
from typeguard import typechecked

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike


class AdaBelief(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Adabelief.

    See paper [AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468).
    """

    @typechecked
    def __init__(
            self,
            learning_rate: Union[FloatTensorLike, Callable] = 0.001,
            beta_1: FloatTensorLike = 0.9,
            beta_2: FloatTensorLike = 0.999,
            epsilon: FloatTensorLike = 1e-8,
            weight_decay_rate: FloatTensorLike = 0.0,
            exclude_from_weight_decay: Optional[List[str]] = None,
            name: str = "AdaBelief",
            **kwargs
    ):
        """Construct a new AdaBelief optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 1st moment estimates.
            beta_2: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay_rate: weight decay rate.
            exclude_from_weight_decay: List of regex patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.
            name: Optional name for the operations created when applying
              gradients. Defaults to "AdaBelief".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
              `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
              norm; `clipvalue` is clip gradients by value, `decay` is
              included for backward compatibility to allow time inverse
              decay of learning rate. `lr` is included for backward
              compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper("weight_decay_rate", weight_decay_rate)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        # This is learning rate decay for using keras learning rate schedule.
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        weight_decay_rate = tf.identity(self._get_hyper("weight_decay_rate", var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m * coefficients["beta_1_t"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * coefficients["one_minus_beta_2_t"]
        v_t = v * coefficients["beta_2_t"] + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        var_update = var - coefficients["lr_t"] * update
        return var.assign(var_update, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # raise NotImplementedError
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m.assign(m * coefficients["beta_1_t"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        print(m_t)
        gm2 = tf.square(tf.math.subtract(grad, m))
        v_scaled_g_values = gm2 * coefficients["one_minus_beta_2_t"]
        v_t = v.assign(v * coefficients["beta_2_t"], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        var_update = var.assign_sub(
            coefficients["lr_t"] * update, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay_rate": self._serialize_hyperparameter(
                    "weight_decay_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
            }
        )
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name




class AdaBeliefTheirs(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Adam algorithm.
    References:
    Adam - A Method for Stochastic Optimization:
      [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)
      ([pdf](https://arxiv.org/pdf/1412.6980.pdf))
    """

    def __init__(self,
                 learning_rate: Union[FloatTensorLike, Callable] = 0.001,
                 beta1: FloatTensorLike = 0.9,
                 beta2: FloatTensorLike = 0.999,
                 epsilon: FloatTensorLike = 1e-8,
                 name="AdaBelief",
                 **kwargs):
        r"""Construct a new Adam optimizer.
        Initialization:
        $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
        $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
        $$t := 0 \text{(Initialize timestep)}$$
        The update rule for `variable` with gradient `g` uses an optimization
        described at the end of section 2 of the paper:
        $$t := t + 1$$
        $$\text{lr}_t := \mathrm{learning_rate} *
          \sqrt{1 - \beta_2^t} / (1 - \beta_1^t)$$
        $$m_t := \beta_1 * m_{t-1} + (1 - \beta_1) * g$$
        $$v_t := \beta_2 * v_{t-1} + (1 - \beta_2) * g * g$$
        $$\text{variable} := \text{variable} -
          \text{lr}_t * m_t / (\sqrt{v_t} + \epsilon)$$
        The default value of 1e-8 for epsilon might not be a good default in
        general. For example, when training an Inception network on ImageNet a
        current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
        formulation just before Section 2.1 of the Kingma and Ba paper rather than
        the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
        hat" in the paper.
        The sparse implementation of this algorithm (used when the gradient is an
        IndexedSlices object, typically because of `tf.gather` or an embedding
        lookup in the forward pass) does apply momentum to variable slices even if
        they were not used in the forward pass (meaning they have a gradient equal
        to zero). Momentum decay (beta1) is also applied to the entire momentum
        accumulator. This means that the sparse behavior is equivalent to the dense
        behavior (in contrast to some momentum implementations which ignore momentum
        unless a variable slice was actually used).
        Args:
          learning_rate: A Tensor or a floating point value.  The learning rate.
          beta1: A float value or a constant float tensor. The exponential decay
            rate for the 1st moment estimates.
          beta2: A float value or a constant float tensor. The exponential decay
            rate for the 2nd moment estimates.
          epsilon: A small constant for numerical stability. This epsilon is
            "epsilon hat" in the Kingma and Ba paper (in the formula just before
            Section 2.1), not the epsilon in Algorithm 1 of the paper.
          use_locking: If True use locks for update operations.
          name: Optional name for the operations created when applying gradients.
            Defaults to "Adam".
        @compatibility(eager)
        When eager execution is enabled, `learning_rate`, `beta1`, `beta2`, and
        `epsilon` can each be a callable that takes no arguments and returns the
        actual value to use. This can be useful for changing these values across
        different invocations of optimizer functions.
        @end_compatibility
        """

        super().__init__(name, **kwargs)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon
        #self.amsgrad = amsgrad

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _get_beta_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return (self._get_non_slot_variable("beta1_power", graph=graph),
                    self._get_non_slot_variable("beta2_power", graph=graph))

    def _create_slots(self, var_list):
        # Create the beta1 and beta2 accumulators on the same device as the first
        # variable. Sort the var_list to make sure this device is consistent across
        # workers (these need to go on the same PS, otherwise some updates are
        # silently ignored).
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(
            initial_value=self._beta1, name="beta1_power", colocate_with=first_var)
        self._create_non_slot_variable(
            initial_value=self._beta2, name="beta2_power", colocate_with=first_var)

        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        lr = self._call_if_callable(self._lr)
        beta1 = self._call_if_callable(self._beta1)
        beta2 = self._call_if_callable(self._beta2)
        epsilon = self._call_if_callable(self._epsilon)

        self._lr_t = ops.convert_to_tensor(lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(epsilon, name="epsilon")

    def _apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)

        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values + epsilon_t, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * step_size

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        graph = None if context.executing_eagerly() else ops.get_default_graph()
        beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values + epsilon_t, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        # Compute the bounds
        step_size = step_size / (v_sqrt + epsilon_t)
        bounded_lr = m_t * step_size

        var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad - m_t) * (grad - m_t) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values + epsilon_t)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        if self.amsgrad:
            vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
            v_sqrt = math_ops.sqrt(vhat_t)
        else:
            vhat_t = state_ops.assign(vhat, vhat)
            v_sqrt = math_ops.sqrt(v_t)

        var_update = state_ops.assign_sub(
            var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values,
            var,
            grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x,
                i,
                v,
                use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(grad, var, indices,
                                         self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            beta1_power, beta2_power = self._get_beta_accumulators()
            with ops.colocate_with(beta1_power):
                update_beta1 = beta1_power.assign(
                    beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = beta2_power.assign(
                    beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(
            *update_ops + [update_beta1, update_beta2], name=name_scope)



    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay_rate": self._serialize_hyperparameter(
                    "weight_decay_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta1": self._serialize_hyperparameter("beta1"),
                "beta2": self._serialize_hyperparameter("beta2"),
                "epsilon": self.epsilon,
            }
        )
        return config
