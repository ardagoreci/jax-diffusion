"""
Defines the models including the U-Net architecture that will be used for gradual denoising
of the latent space.
"""
import jax
import flax
from nn import normalization
from flax import linen as nn
import jax.numpy as jnp
from abc import abstractmethod


class AttentionPool2d(nn.Module):
    spacial_dim: int
    embed_dim: int
    num_heads_channels: int
    output_dim: int = None

    def setup(self):
        # TODO: implement initialization
        pass

    def __call__(self, x):
        # TODO: implement attention pooling
        pass


class TimeStepBlock(nn.Module):
    """
    Any module where __call__ takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def __call__(self, x, emb):
        """
        Apply the module to 'x' given 'emb' as the timestep embedding.
        """


class TimeStepEmbedSequential(nn.Sequential, TimeStepBlock):
    """
    A sequential module that passes timestep embeddings to children
    that support it (instances of TimeStepBlock) as an extra input.
    """

    def __call__(self, x, emb):
        # TODO: check if this is compatible with Flax Sequential
        for layer in self:
            if isinstance(layer, TimeStepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling module with an optional convolution.

    channels: channels in the inputs and outputs
    use_conv: a bool determining if convolution is applied
    dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
          upsampling occurs in the inner-two dimensions.

    (unit-tested)
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None

    @nn.compact
    def __call__(self, x):
        if self.dims == 3:
            (B, D, H, W, C) = x.shape
            x = jax.image.resize(x, shape=(B, D, H * 2, W * 2, C),
                                 method='nearest')
        elif self.dims == 2:
            (B, H, W, C) = x.shape
            x = jax.image.resize(x, shape=(B, H * 2, W * 2, C),
                                 method='nearest')
        elif self.dims == 1:
            (B, H, C) = x.shape
            x = jax.image.resize(x, shape=(B, H * 2, C), method='nearest')
        else:
            raise ValueError(f"Unsupported dimensions: {self.dims}")

        if self.use_conv:
            channels = self.channels
            if self.out_channels is not None:
                channels = self.out_channels  # set the output channels if they are not equal to self.channels

            kernel_length = 3
            kernel_size = tuple([kernel_length for i in range(self.dims)])
            x = nn.Conv(features=channels,
                        kernel_size=kernel_size,
                        strides=1,
                        padding='SAME')(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling module with an optional convolution.

    channels: channels in the inputs and outputs
    use_conv: a bool determining if convolution is applied
    dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
          downsampling occurs in the inner-two dimensions.

    (if use_conv, the method performs a convolutional downsampling using strides.
    else, it performs an average pooling downsampling.)

    (unit-tested)
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None

    @nn.compact
    def __call__(self, x):
        # Determine the number of output channels.
        if self.out_channels is not None:
            out_channels = self.out_channels
        else:
            out_channels = self.channels

        # If use_conv, use convolution with stride 2.
        if self.use_conv:
            kernel_size = tuple([3 for i in range(self.dims)])  # 3x3 convolution for 2D, 3x3x3 convolution for 3D
            # There is an edge-case with 3D convolutions.
            if self.dims == 3:
                x = nn.Conv(features=out_channels,
                            kernel_size=kernel_size,
                            strides=(1, 2, 2),  # Keeps the spatial dimensions the same
                            padding='SAME')(x)

            elif self.dims == 2 or self.dims == 1:
                x = nn.Conv(features=out_channels,
                            kernel_size=kernel_size,
                            strides=2,
                            padding='SAME')(x)
            else:
                raise ValueError(f"Unsupported dimensions: {self.dims}")

        else:
            if self.dims == 1:
                x = nn.avg_pool(x, window_shape=(2,), strides=(2,))
            elif self.dims == 2:
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
            elif self.dims == 3:
                x = nn.avg_pool(x,
                                window_shape=(1, 2, 2),
                                strides=(1, 2, 2),
                                padding='SAME')

        return x


class ResBlock(TimeStepBlock):
    """
    A residual block that can optionally change the number of channels.

    channels: the number of input channels
    emb_channels: the number of timestep embedding channels
    use_conv: if True and out_channels is specified, use a spatial
              convolution instead of a smaller 1x1 convolution to change the
              channels in the skip connection.
    out_channels: if specified, the number of out channels
    dims: determines if the signal is 1D, 2D, or 3D.
    use_checkpoint: if True, use gradient checkpointing on this module
    up: if True, use this block for upsampling
    down: if True, use this block for downsampling
    """
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    use_conv: bool = True
    dims: int = 2
    use_scale_shift_norm: bool = False
    up: bool = False
    down: bool = False

    def setup(self):
        self.out_channels = self.out_channels or self.channels  # TODO: this will not be allowed in Flax!

        self.in_layers = nn.Sequential([
            # TODO: implement the input layers here
            # normalization(channels),
            # nn.SiLU(),
            # conv_nd(dims, channels, self.out_channels, 3, padding=1)
        ])

        self.updown = self.up or self.down  # TODO: be careful with this, don't want any bugs
        # TODO: (1) initialize the upsampling or downsampling layers
        # TODO: (2) initialize the embedding layers
        # TODO: (3) initialize the output layers

    def __call__(self, x, emb):
        # TODO: implement the forward pass of ResBlock
        pass


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    channels: the number of input channels
    num_heads: the number of attention heads
    num_head_channels:
    use_new_attention_order:

    """
    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_new_attention_order: bool = False

    @nn.compact
    def __call__(self, x):
        B, *spatial, C = x.shape  # get dimensions
        # flatten x
        x = jnp.reshape(x, (B, -1, C))
        # normalize
        h = normalization(C)(x)
        # convolve to get qkv
        qkv = nn.Conv(features=self.num_heads * self.channels * 3,  # Getting the qkv tensor for multi-head attention
                      kernel_size=1)(h)
        # attention
        h = QKVAttention(self.num_heads)(qkv)
        # transpose to (B, d_model, C)
        h = h.transpose((0, 2, 1))  # getting channels last for convolutions
        # projection layer
        h = (nn.Conv(features=self.channels, kernel_size=1))(h)  # TODO: there is supposed to be a ZeroModule here
        # Ho et al. have not employed such a zero module, I suspect it is to make this layer a purely skip connection
        # at initialization - only later do the layers diverge from zero.

        # residual connection and reshape to original shape
        return (x + h).reshape(B, *spatial, C)


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """
    num_heads: int

    @nn.compact
    def __call__(self, q, k, v):
        pass


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.

    Note: I do not understand the need for two different attention modules.
    This might be avoided upstream by using better array handling.

    TODO: this class can only handle num_heads = 1 for now!
    The issue is that I have not found a way to get the value vector to be of shape
    (H, T, d_model / H) for H > 1. If this is the case, the class will return a
    value tensor that is enlarged to (T, d_model*H) - this may cause problems downstream.
    """
    num_heads: int = 1

    @nn.compact
    def __call__(self, qkv):
        """
        Args:
            qkv: the combined query key value tensor of shape (B, d_model, num_heads * T * 3)
        Returns:
            a value tensor of shape (B, T, d_model)
        """
        # Shape wrangling
        B, d_model, product = qkv.shape
        T = product // (self.num_heads * 3)
        qkv = qkv.reshape((B, d_model, self.num_heads, T, 3))  # (B, d_model, num_heads, T, 3)
        qkv = qkv.transpose((0, 4, 2, 3, 1))  # (B, 3, num_heads, T, d_model)
        q, k, v = jnp.split(qkv, 3, axis=1)
        q = jnp.squeeze(q, axis=1)  # remove unnecessary dimension
        k = jnp.squeeze(k, axis=1)
        v = jnp.squeeze(v, axis=1)

        # Attention
        return jax.vmap(self._forward)(q, k, v)

    def _forward(self, q, k, v) -> jnp.ndarray:
        """
        Args:
            q: the query tensor (H, T, d_model)
            k: the key tensor (H, T, d_model)
            v: the value tensor (H, T, d_model / H)
        where H is the number of heads, T is the number of elements that attend to each other
        and C is d_model
        Returns: a value tensor of shape (T, d_model) after attention

        This function does not take into account the batch dimension. It will be vmap
        transformed to do so.
        (unit-tested)
        """
        headed_v = jax.vmap(self.scaled_dot_product_attention)(q, k, v)  # (H, T, d_model/num_heads)
        # Merge the heads to recover v with dims (T, d_model).
        v_prime = jnp.concatenate(headed_v, axis=-1)  # (T, d_model)
        return v_prime

    @staticmethod
    def scaled_dot_product_attention(q, k, v) -> jnp.ndarray:
        """
        Args:
            q: the query tensor (T, d_model)
            k: the key tensor (T, d_model)
            v: the value tensor (T, d_model / num_heads)
        Returns: a tensor of shape (T, d_model / num_heads) after scaled dot product attention

        This function will be vmap transformed in order to apply it as a multi-head
        attention mechanism.
        (unit-tested)
        """
        # Get dimensions.
        elements, d_model = q.shape
        scale_factor = jnp.sqrt(d_model)  # scaling factor for scaled attention
        weight_logits = jnp.matmul(q, k.T) / scale_factor  # Compute Q*K.T / sqrt(d_model)
        attention_weights = jax.nn.softmax(weight_logits, axis=-1)  # (T, T)
        attention_v = jnp.matmul(attention_weights, v)  # (T, d_model / num_heads)
        return attention_v
