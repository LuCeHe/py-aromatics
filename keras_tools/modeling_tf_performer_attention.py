"""
sources:
https://github.com/norabelrose/transformers-plus-performers/
"""

import logging
import math
import random
import tensorflow as tf

from typing import Callable, Optional, Union

from pyaromatics.keras_tools.configuration_performer_attention import PerformerKernel, \
    PerformerAttentionConfig, OrthogonalFeatureAlgorithm

KERNEL_CALLABLES = {
    PerformerKernel.cosh: lambda x, h: tf.concat((tf.exp(h + x), tf.exp(h - x)), axis=-1),
    PerformerKernel.exp: lambda x, h: tf.exp(h + x),  # Default
    PerformerKernel.elu: lambda x: tf.nn.elu(x) + 1,
    PerformerKernel.relu: tf.nn.relu
}


def resolve_enum(enum_class, value):
    return enum_class[value] if isinstance(value, str) else value


class TFPerformerAttention(tf.keras.layers.Layer):
    def __init__(self, config: Optional[Union[dict, PerformerAttentionConfig]] = None, **kwargs):
        super().__init__(name=kwargs.pop('name', None), dtype=kwargs.pop('dtype', None))
        self.config = config
        if isinstance(config, dict):
            config = PerformerAttentionConfig(**config)
        else:
            config = config or PerformerAttentionConfig()

        # kwargs take precedence over the default values that might be stored in the config object
        for k, v in kwargs.items():
            assert hasattr(config, k), f"'{k}' is an invalid config parameter"
            setattr(config, k, v)

        self.__dict__.update(config.__dict__)

        assert self.num_heads and self.d_model, "Num_heads and d_model must be non-None"
        assert self.d_model % self.num_heads == 0, "Num_heads must divide d_model evenly"
        assert self.d_model > self.num_heads, "Number of dimensions per head must be greater than 1"

        self.dropout = tf.keras.layers.Dropout(rate=self.attention_dropout)
        self.calls_since_last_redraw = 0

        self.orthogonal_feature_algorithm = resolve_enum(OrthogonalFeatureAlgorithm, self.orthogonal_feature_algorithm)
        assert self.orthogonal_feature_algorithm != OrthogonalFeatureAlgorithm.kacs, \
            "Kac's random walk is not supported in TensorFlow"

        # Create the feature matrix up front if we don't need to know what the batch dimension is;
        # otherwise, lazily create it on the first forward pass
        self.random_features = None
        if not self.use_thick_features:
            self._generate_feature_matrix(batch_size=1)

        # Recurrent state
        if self.use_recurrent_decoding:
            self.s = None
            self.z = None

        if isinstance(self.kernel_type, Callable):
            self.kernel_fn = self.kernel_type  # Allow for custom kernel types
        else:
            self.kernel_type = resolve_enum(PerformerKernel, self.kernel_type)
            self.kernel_fn = KERNEL_CALLABLES[self.kernel_type]

        if self.use_linear_layers:
            for name in self.linear_layer_names:
                setattr(self, name, tf.keras.layers.Dense(units=self.d_model))

    def prune_heads(self, heads):
        raise NotImplementedError

    def redraw_features_now(self):
        """
        Immediately redraws the random features.
        """
        batch = self.random_features.shape[0]
        self._generate_feature_matrix(batch)

        if self.redraw_verbose:
            logging.getLogger().info("TFPerformerAttention: Just redrew random features.")

        self.calls_since_last_redraw = 0

    def reset_recurrent_state(self):
        """
        Resets the recurrent state kept by the model when use_recurrent_decoding == True
        """
        self.s = None
        self.z = None

    def call(self, inputs, mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: tf.tensor(bs, num_heads, seq_length, seq_length) Attention weights context: tf.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        query, key, value = inputs
        bs, q_length, dim = query.shape
        dim_per_head = self.d_model // self.num_heads

        def shape(x):
            """ separate heads """
            r = tf.reshape(x, (-1, tf.shape(x)[1], self.num_heads, dim_per_head))
            return tf.transpose(r, perm=[0, 2, 1, 3])

        if self.use_linear_layers:
            query, key, value = (getattr(self, name)(x) for name, x in
                                 zip(self.linear_layer_names, (query, key, value)))

        # (bs, num_heads, q_length, dim_per_head)
        query, key, value = (shape(x) for x in (query, key, value))

        assert not output_attentions, "Can't output attention maps when using Performer attention."
        if self.use_recurrent_decoding:
            assert q_length == 1, "When use_recurrent_decoding == True, we only input and output one token at a time."

        self._redraw_features_if_needed(bs)

        # Get the transformed values of Q and K
        q_prime, k_prime = self.get_projected_queries_and_keys(query, key)
        return self.compute_attention_with_projected_queries_and_keys(q_prime, k_prime, value, mask, head_mask)

    def get_projected_queries_and_keys(self, q, k):
        """
        Turns Q into Q' and K into K' by multiplying them by the random feature tensor.
        Parameters:
            q: torch.tensor(bs, seq_length, dim)
            k: torch.tensor(bs, seq_length, dim)

        Returns:
            q_prime: torch.tensor(bs, seq_length, num_features)
            k_prime: torch.tensor(bs, seq_length, num_features)
        """
        # Instead of dividing the product QK^T by sqrt(d), we divide Q and K by the 4th root of d.
        q = q / (self.d_model ** 0.25)
        k = k / (self.d_model ** 0.25)

        projected_q = q @ self.random_features
        projected_k = k @ self.random_features

        # Special logic for kernels that attempt to approximate softmax
        if self.kernel_type in (PerformerKernel.cosh, PerformerKernel.exp):
            # The h(x) function is defined in Lemma 1 in Choromanski et al. pg. 4 as exp(-||x||**2 / 2). For numerical
            # stability we leverage the fact that exp(x)*exp(y) = exp(x + y) here and delay computing the exp().
            h_of_q = -tf.math.reduce_sum(q ** 2, axis=-1, keepdims=True) / 2
            h_of_k = -tf.math.reduce_sum(k ** 2, axis=-1, keepdims=True) / 2

            # Compute the numerical stabilizer that we subtract from the input to exp(). For some reason the original
            # Jax implementation uses different types of stabilizers for queries vs. keys, and we follow that here.
            q_stabilizer = tf.math.reduce_max(h_of_q, axis=-1, keepdims=True)

            # This is just a scalar
            k_stabilizer = tf.math.reduce_max(h_of_k)

            q_kernel_output = self.kernel_fn(projected_q - q_stabilizer, h_of_q)
            k_kernel_output = self.kernel_fn(projected_k - k_stabilizer, h_of_k)

            # By multiplying by 1/sqrt(m), we ensure the final matrix product will contain a factor of 1/m. This means
            # each row of Q'K'^T can be interpreted as an average over the exp(omega^T * q) * exp(omega^T * k) terms.
            normalizing_constant = (q_kernel_output.shape[-1] ** -0.5)

            q_prime = normalizing_constant * (q_kernel_output + self.kernel_epsilon)
            k_prime = normalizing_constant * (k_kernel_output + self.kernel_epsilon)
            return q_prime, k_prime

        # Generalized attention (ReLU, ELU...)
        else:
            return tuple(self.kernel_fn(x) + self.kernel_epsilon for x in (projected_q, projected_k))

    def compute_attention_with_projected_queries_and_keys(self, q_prime, k_prime, v, mask=None, head_mask=None):
        """
        Computes the attention output given Q' and K' from the above get_projected_queries_and_keys method.
        Parameters:
            q_prime: tf.tensor(bs, seq_length, num_features)
            k_prime: tf.tensor(bs, seq_length, num_features)
            v: tf.tensor(bs, seq_length, dim)
            mask: tf.tensor(bs, seq_length)

        Returns:
            V': tf.tensor(bs, seq_length, dim)
        """
        # Apply the padding mask to K'. Also applying it to Q' would be redundant.
        if mask is not None:
            k_prime *= tf.expand_dims(tf.expand_dims(mask, 1), -1)

        k_prime_t = tf.linalg.matrix_transpose(k_prime)
        output = self._numerator_for_projected_queries_and_keys(q_prime, k_prime_t, v)

        if self.normalize_output:
            output /= self._denominator_for_projected_queries_and_keys(q_prime, k_prime_t)

        return self._finalize_attention_output(output, head_mask)

    def _numerator_for_projected_queries_and_keys(self, q_prime, k_prime_t, v):
        # Noncausal
        if not self.causal:
            output = q_prime @ (k_prime_t @ v)

        elif not self.use_recurrent_decoding:
            # Causal, during training
            output = _headwise_causal_numerator(q_prime, k_prime_t, v)

        else:
            # Causal, at inference time
            s_delta = k_prime_t @ v
            self.s = s_delta if self.s is None else self.s + s_delta
            output = q_prime @ self.s

        return output

    def _denominator_for_projected_queries_and_keys(self, q_prime, k_prime_t):
        # Noncausal
        if not self.causal:
            denom = q_prime @ tf.math.reduce_sum(k_prime_t, axis=-1, keepdims=True)  # Sum over positions

        # Causal, during training
        elif not self.use_recurrent_decoding:
            prefix_sums = tf.cumsum(k_prime_t, axis=-1)  # Cumsum over positions
            denom = tf.einsum("bhlm,bhml->bhl", q_prime, prefix_sums)
            denom = tf.expand_dims(denom, axis=-1)

        # Causal, at inference time
        else:
            self.z = k_prime_t if self.z is None else self.z + k_prime_t  # Incrementally sum over positions
            denom = q_prime @ self.z

        # Avoid dividing by very small numbers
        extreme_vals = tf.cast(tf.math.abs(denom) <= self.normalization_stabilizer, denom.dtype)
        return denom + 2 * self.normalization_stabilizer * extreme_vals

    def _finalize_attention_output(self, context, head_mask=None, att_map_to_output=None):
        # Mask heads if we want to
        if head_mask is not None:
            context = context * head_mask

        x = tf.transpose(context, perm=[0, 2, 1, 3])  # [...seq_len, num_heads, dim_per_head]
        context = tf.reshape(x, (-1, tf.shape(x)[1], self.d_model))  # (bs, q_length, dim)

        if self.use_linear_layers and len(self.linear_layer_names) > 3:
            context = getattr(self, self.linear_layer_names[3])(context)  # (bs, q_length, dim)

        if att_map_to_output:
            return context, att_map_to_output
        else:
            return context,

    def _generate_feature_matrix(self, batch_size):
        dim_per_head = self.d_model // self.num_heads
        num_rows = int(self.num_random_features or round(dim_per_head * math.log(dim_per_head)))
        batch = batch_size if self.use_thick_features else 1

        if not self.use_orthogonal_features:
            print(batch, num_rows, dim_per_head)
            final_tensor = tf.random.normal((batch, num_rows, dim_per_head))
        else:
            total_num_blocks = int(math.ceil(num_rows / dim_per_head))
            extra_rows = total_num_blocks * dim_per_head - num_rows

            blocks = [_get_orthogonal_block(batch, dim_per_head) for _ in range(total_num_blocks)]
            if extra_rows > 0:
                blocks[-1] = blocks[-1][:, extra_rows:]

            final_tensor = tf.concat(blocks, axis=1)

            # This option yields SMREG
            if self.regularize_feature_norms:
                final_tensor *= dim_per_head ** 0.5
            else:
                # Hack to make the matrix columns have the norm we would expect them to have if they were sampled
                # straight from a Gaussian, instead of being all norm 1 since they went through QR decomposition
                multiplier = tf.norm(tf.random.normal((batch, num_rows, dim_per_head)), axis=-1)
                final_tensor = tf.linalg.diag(multiplier) @ final_tensor

        final_tensor = tf.expand_dims(final_tensor, axis=1)  # Add an attention head dimension
        final_tensor = tf.linalg.matrix_transpose(final_tensor)
        self.random_features = final_tensor

    def _redraw_features_if_needed(self, batch):
        # We haven't created the projection matrix yet, let's create it
        if self.random_features is None:
            self._generate_feature_matrix(batch)

        elif self.feature_redraw_interval is not None:
            if self.redraw_stochastically:
                # random.random() returns a float between 0.0 and 1.0, so this expression
                # evaluates to True with probability 1. / self.feature_redraw_interval
                if random.random() < 1. / self.feature_redraw_interval:
                    self.redraw_features_now()

            # It's time to redraw the projection matrix
            elif self.calls_since_last_redraw >= self.feature_redraw_interval:
                self.redraw_features_now()

            # Keep track of how many forward passes we do before we redraw again
            else:
                self.calls_since_last_redraw += 1


    def get_config(self):
        config = {
            'config': self.config.to_dict(),
        }
        return dict(list(super().get_config().items()) + list(config.items()))


def _get_orthogonal_block(batch, size):
    with tf.device('/CPU:0'):
        unstructured_block = tf.random.normal((batch, size, size))
        orthog, r = tf.linalg.qr(unstructured_block)

    return tf.linalg.matrix_transpose(orthog)


def _headwise_causal_numerator(q_prime, k_prime_t, v):
    results = []

    # Iterate over the attention heads to avoid allocating a very large tensor
    for head in range(q_prime.shape[1]):
        # Outer products- a sorta biggish tensor
        outer_prods = tf.einsum('bml,bld->blmd', k_prime_t[:, head], v[:, head])
        prefix_sums = tf.cumsum(outer_prods, axis=1)

        query_prods = tf.einsum('blmd,blm->bld', prefix_sums, q_prime[:, head])
        results.append(tf.expand_dims(query_prods, axis=1))

    return tf.concat(results, axis=1)


if __name__ == '__main__':
    bs, sl, dm = 32, 300, 32
    nh = 2
    x = tf.random.normal((bs, sl, dm))

    config = PerformerAttentionConfig
    config.d_model = dm
    config.num_heads = nh
    config.causal = True

    # ---------------------------------------------------
    import time

    tf.random.set_seed(0)
    config.causal = True
    performer = TFPerformerAttention(config)
    start_time = time.time()
    o, = performer([x, x, x], mask=None, head_mask=None, output_attentions=False)

    print(o.shape)
    # print(o)
    print("Non recurrent took {}s".format(time.time() - start_time))
    # ---------------------------------------------------
    tf.random.set_seed(0)
    config.use_recurrent_decoding = True
    performer = TFPerformerAttention(config)
    start_time = time.time()
    os = []
    for t in range(sl):
        x_t = x[:, t][:, None]
        o_t, = performer([x_t, x_t, x_t], mask=None, head_mask=None, output_attentions=False)
        os.append(o_t)
    o = tf.concat(os, axis=1)

    # o, = performer(x, x, x, mask=None, head_mask=None, output_attentions=False)

    print(o.shape)
    # print(o)
    print("Recurrent took {}s".format(time.time() - start_time))
