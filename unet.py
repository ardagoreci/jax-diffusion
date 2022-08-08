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
    """
    channels: int
    use_conv: bool
    dims: int = 2
    out_channels: int = None

    def setup(self):
        if self.use_conv:
            # self.conv = nn.Conv(channels, channels, 3, 1, 1)
            # TODO: implement convolutional upsampling
            pass
        self.out_channels = self.out_channels or self.channels

    def __call__(self, x):
        # TODO: implement the interpolation bit here
        if self.use_conv:
            x = self.conv(x)
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

    def setup(self):
        self.op = None
        if self.use_conv:
            # TODO: self.op = conv(...)
            pass
        else:
            # TODO: self.op = avg_pool(...)
            pass

    def __call__(self, x):
        return self.op(x)


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
        self.out_channels = self.out_channels or self.channels

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
