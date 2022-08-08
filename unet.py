"""
Defines the models including the U-Net architecture that will be used for gradual denoising
of the latent space.
"""
import jax
import flax
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
        # TODO: implement the interpolation bit here

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
            # TODO: not sure if this method can be handled within Flax.
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
            kernel_size = tuple([3 for i in range(self.dims)])
            x = nn.Conv(features=out_channels,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='SAME')(x)
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

    channels:
    num_heads: the number of attention heads
    num_head_channels:
    use_new_attention_order:
    """
    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_new_attention_order: bool = False

    def setup(self):
        if self.num_head_channels != -1:
            # TODO: assert divisibility here
            self.num_heads = self.channels // self.num_head_channels
        # TODO: initialize normalization layers
        # self.norm = normalization(self.channels)

        if self.use_new_attention_order:
            # TODO: initialize QKVAttention(self.num_heads)
            pass
        else:
            # TODO: initialize QKVAttentionLegacy(self.num_heads)
            pass
        # TODO: initialize the self.proj_out layer (don't know yet)

    def __call__(self, x):
        # TODO: implement the forward pass of the attention block
        pass


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    num_heads: int

    @nn.compact
    def __call__(self, qkv):
        pass


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    num_heads: int

    @nn.compact
    def __call__(self, qkv):
        pass
