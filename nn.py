"""
Various utilities for neural networks.
"""
from flax import linen as nn


def normalization(channels):
    """
    Create a normalization layer.
    TODO: unit-test this layer. (I don't know what it is doing.)
    """
    return nn.GroupNorm(num_groups=None, group_size=channels)


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolutional module.
    """
    if dims == 1:
        # TODO: define a 1D convolutional module
        pass
    elif dims == 2:
        # TODO: define a 2D convolutional module
        pass
    elif dims == 3:
        # TODO: define a 3D convolutional module
        pass
    else:
        raise ValueError(f"Unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        # TODO: define a 1D average pooling module
        pass
    elif dims == 2:
        # TODO: define a 2D average pooling module
        pass
    elif dims == 3:
        # TODO: define a 3D average pooling module
        pass
    raise ValueError(f"Unsupported dimensions: {dims}")


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
    # TODO: implement timestep_embeddings
    pass
