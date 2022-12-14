"""
Utilities for neural networks.
"""
from flax import linen as nn
import jax
import jax.numpy as jnp


def normalization(channels):
    """
    Create a normalization layer.
    TODO: unit-test this layer. (I don't know what it is doing.)
    """
    return nn.GroupNorm(num_groups=None, group_size=channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    Args:
        timesteps: a 1-D Tensor of N indices, one per batch element.
        dim: the dimension of the output
        max_period: controls the minimum frequency of the embeddings.
    Returns:
        a Tensor of shape (N, dim) of positional embeddings
    """
    half = dim // 2
    # Compute freqs with the formula:
    # freqs = exp(-log(max_period) * arange(0, half) / half)
    # args = weird bit of logic here
    # embedding = concat([cos(args), sin(args)], axis=-1)
    # if dim % 2 == 0:
    #    embedding = concat([embedding, zeros_like(embedding[:, :1]), dims=1)
    # return embedding
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(0, half, dtype=jnp.float32) / half)
    args = timesteps[:, None] * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    if dim % 2:
        embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
    return embedding
