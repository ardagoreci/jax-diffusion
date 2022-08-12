"""
Defines the models including the U-Net architecture that will be used for gradual denoising
of the latent space.
"""
import jax
import flax
from nn import normalization, timestep_embedding
from flax import linen as nn
import jax.numpy as jnp
from abc import abstractmethod
from typing import Collection


class AttentionPool2d(nn.Module):
    spacial_dim: int
    embed_dim: int
    num_head_channels: int
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
        for layer in self.layers:
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


class Identity(nn.Module):
    """
    A utility module that simply returns the input.
    """
    @nn.compact
    def __call__(self, x):
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
    use_scale_shift_norm: # TODO: not implemented yet!
    up: if True, use this block for upsampling
    down: if True, use this block for downsampling

    TODO: the current version of the ResBlock does not handle scale_shift_norm

    (unit-tested)
    """
    channels: int
    emb_channels: int
    dropout: float = 0.0
    out_channels: int = None
    use_conv: bool = True
    dims: int = 2
    use_scale_shift_norm: bool = False
    up: bool = False
    down: bool = False

    def setup(self):
        out_channels = self.out_channels or self.channels

        # in_rest and in_conv form the in_layers, but they need to be separate
        self.in_rest = nn.Sequential([
            normalization(self.channels),
            jax.nn.silu,
        ])
        self.in_conv = nn.Conv(features=out_channels,
                               kernel_size=tuple([3 for i in range(self.dims)]),
                               padding='SAME')
        self.updown = self.up or self.down  # whether to upsample or downsample
        # (1) initialize the upsampling or downsampling layers
        if self.up:
            self.h_upd = Upsample(self.channels, use_conv=False, dims=self.dims)
            self.x_upd = Upsample(self.channels, use_conv=False, dims=self.dims)
        elif self.down:
            self.h_upd = Downsample(self.channels, use_conv=False, dims=self.dims)
            self.x_upd = Downsample(self.channels, use_conv=False, dims=self.dims)
        else:
            self.h_upd = Identity()
            self.x_upd = Identity()

        # (2) initialize the embedding layers
        self.emb_layers = nn.Sequential([
            jax.nn.silu,
            nn.Dense(features=(
                2 * out_channels if self.use_scale_shift_norm else out_channels)
            )
        ])

        # (3) initialize the output layers
        self.out_layers = nn.Sequential([
            normalization(out_channels),
            jax.nn.silu,
            nn.Dropout(self.dropout),
            nn.Conv(features=out_channels,
                    kernel_size=tuple([3 for i in range(self.dims)]),
                    padding='SAME')
        ])

        # Skip connection logic
        if out_channels == self.channels:
            self.skip_connection = Identity()
        elif self.use_conv:
            # self.skip_connection = nn.Conv()
            self.skip_connection = nn.Conv(features=out_channels,
                                           kernel_size=tuple([3 for i in range(self.dims)]),
                                           padding='SAME')
        else:
            # If not using convolution, use a 1x1 conv to get the output channels right.
            self.skip_connection = nn.Conv(features=out_channels,
                                           kernel_size=tuple([1 for i in range(self.dims)]),
                                           padding='SAME')

    def __call__(self, x, emb):
        # (1) if up or downsample, use the upsampling or downsampling layers (apply the convolution in in_layers after
        # the op). Otherwise, apply in_rest and in_conv directly.
        if self.updown:
            h = self.in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.in_conv(h)
        else:
            h = self.in_rest(x)
            h = self.in_conv(h)

        # (2) embedding layers logic
        emb_out = self.emb_layers(emb)

        # (3) scale shift norm ? (else, add the timestep embedding and pass it through the output layers)
        if self.use_scale_shift_norm:
            raise NotImplementedError("scale-shift normalization not implemented")
        else:
            h = h + emb_out
            h = self.out_layers(h)
        # (4) residual with the skip connection, return sum.
        return self.skip_connection(x) + h

    class UNetModel(nn.Module):
        """
        The full UNet model with attention and timestep embedding.
        in_channels: channels in the input Tensor.
        model_channels: base channel count for the model.
        out_channels: channels in the output Tensor.
        num_res_blocks: number of residual blocks per downsample.
        attention_resolutions: a collection of downsample rates at which
            attention will take place. May be a set, list, or tuple.
            For example, if this contains 4, then at 4x downsampling, attention
            will be used.
        dropout: the dropout probability.
        channel_mult: channel multiplier for each level of the UNet.
        conv_resample: if True, use learned convolutions for upsampling and downsampling.
        dims: determines if the signal is 1D, 2D, or 3D.
        num_heads: the number of attention heads in each attention layer.
        num_head_channels: if specified, ignore num_heads and instead use
                            a fixed channel width per attention head.
        num_heads_upsample: works with num_heads to set a different number
                            of heads for upsampling. Deprecated.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.
        resblock_updown: use residual blocks for up/downsampling.
        use_new_attention_order: use a different attention pattern for potentially
                                 increased efficiency.

        Deleted pieces:
        - checkpointing
        - num_classes (not planning on doing class-conditional generation)
        """
        in_channels: int
        model_channels: int
        out_channels: int
        num_res_blocks: int
        attention_resolutions: Collection[int]
        dropout: float = 0.0
        channel_mult: Collection[int] = (1, 2, 4, 8)
        conv_resample: bool = True
        dims: int = 2
        num_heads: int = 1
        num_head_channels: int = -1
        num_heads_upsample: int = -1
        use_scale_shift_norm: bool = False
        resblock_updown: bool = False
        use_new_attention_order: bool = False

        def setup(self):
            time_embed_dim = self.model_channels * 4
            self.time_embed = nn.Sequential([
                nn.Dense(time_embed_dim),
                jax.nn.silu,
                nn.Dense(time_embed_dim),
            ])  # projects model_channel to size 4 * model_channel

            ch = input_ch = int(self.channel_mult[0] * self.model_channels)

            # 1st element of input_blocks: a convolution with output channels equal to ch
            self.input_blocks = [nn.Conv(features=ch, kernel_size=tuple([3 for i in range(self.dims)]), padding='SAME')]
            self._feature_size = ch

            # Add the channels to a list input_block_channels
            input_block_channels = [ch]
            ds = 1

            # Enumerate channel_mult to (level, mult)
            #    for _ in range(self.num_res_blocks):
            #       Add a residual block to input blocks
            #       Check attention_resolutions, if this res is in it, add an attention block as well
            #       Add ch = int(mult * model_channels) to input_block_channels
            #    endfor
            #    if level != len(channel_mult) - 1:   (meaning if we are not at the end of the channel_mult list)
            #      add a downsampling residual block, (or a simple Downsample if resblock_updown is False)
            #    update ds, ch, and self.feature_size
            for level, mult in enumerate(self.channel_mult):
                for _ in range(self.num_res_blocks):
                    layers = [
                        ResBlock(channels=ch,
                                 emb_channels=time_embed_dim,
                                 dropout=self.dropout,
                                 out_channels=int(mult * self.model_channels),  # double channels with each halving of
                                 # resolution
                                 )
                    ]
                    ch = int(mult * self.model_channels)  # set channels to the output of ResBlock for the attention
                    # (or next iteration)
                    if ds in self.attention_resolutions:
                        layers.append(
                            AttentionBlock(channels=ch,
                                           num_heads=self.num_heads,
                                           num_head_channels=self.num_head_channels)
                        )
                    self.input_blocks.append(TimeStepEmbedSequential(*layers))  # make everything into a nn.Sequential
                    self._feature_size += ch
                    input_block_channels.append(ch)
                    # ends inner for
                if level != len(self.channel_mult) - 1:
                    out_ch = ch
                    if self.resblock_updown:
                        self.input_blocks.append(
                            ResBlock(channels=ch,
                                     emb_channels=time_embed_dim,
                                     dropout=self.dropout,
                                     out_channels=out_ch,
                                     down=True)
                        )
                    else:
                        self.input_blocks.append(Downsample(channels=ch,
                                                            use_conv=self.conv_resample,
                                                            dims=self.dims,
                                                            out_channels=out_ch))
                    ch = out_ch  # required for the next run of the for loop
                    input_block_channels.append(ch)
                    ds *= 2
                    self._feature_size += ch

            # Middle Block consists of:
            # - ResBlock
            # - AttentionBlock
            # - ResBlock
            self.middle_block = TimeStepEmbedSequential([
                ResBlock(channels=ch,
                         emb_channels=time_embed_dim,
                         dropout=self.dropout,
                         out_channels=out_ch),
                AttentionBlock(channels=ch,
                               num_heads=self.num_heads,
                               num_head_channels=self.num_head_channels),
                ResBlock(channels=ch,
                         emb_channels=time_embed_dim,
                         dropout=self.dropout,
                         out_channels=out_ch)
            ])
            self._feature_size += ch

            # Output Block
            # do the same logic as the input layers, only in reverse
            self.output_blocks = []
            for level, mult in reversed(list(enumerate(self.channel_mult))):
                for i in range(self.num_res_blocks + 1):
                    ich = input_block_channels.pop()
                    layers = [
                        ResBlock(channels=ich + ch,
                                 emb_channels=time_embed_dim,
                                 dropout=self.dropout,
                                 out_channels=int(mult * self.model_channels),
                                 dims=self.dims)
                    ]
                    ch = int(mult * self.model_channels)
                    if ds in self.attention_resolutions:
                        layers.append(AttentionBlock(channels=ch,
                                                     num_heads=self.num_heads,
                                                     num_head_channels=self.num_head_channels))
                    if level and i == self.num_res_blocks:
                        out_ch = ch
                        if self.resblock_updown:
                            layers.append(
                                ResBlock(channels=ch,
                                         emb_channels=time_embed_dim,
                                         dropout=self.dropout,
                                         out_channels=out_ch,
                                         up=True)
                            )
                        else:
                            layers.append(Upsample(channels=ch,
                                                   use_conv=self.conv_resample,
                                                   dims=self.dims,
                                                   out_channels=out_ch))
                        ds //= 2
                    self.output_blocks.append(TimeStepEmbedSequential(*layers))
                    self._feature_size += ch

            # Final Output
            self.out = nn.Sequential([
                normalization(ch),
                jax.nn.silu,
                nn.Conv(features=out_ch,
                        kernel_size=tuple([3 for i in range(self.dims)]),
                        padding='SAME'),
            ])

        def __call__(self, x, timesteps):
            """
            Args:
                x: [N x ... (spatial dims)] Tensor of inputs
                timesteps: a 1D batch of timesteps
            Returns: [N x ... (spatial dims)] Tensor of outputs
            """
            # Required in the symmetric skip connections.
            hs = []
            # Timestep embeddings
            emb = self.time_embed(timestep_embedding(timesteps))  # (N, time_embed_dim)
            h = x
            for module in self.input_blocks:
                h = module(h, emb)
                hs.append(h)
            h = self.middle_block(h, emb)
            for module in self.output_blocks:
                h = jnp.concatenate([h, hs.pop()], axis=1)
                h = module(h, emb)
            return self.out(h)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    channels: the number of input channels
    num_heads: the number of attention heads
    num_head_channels: the number of channels per attention head
    use_new_attention_order:

    """
    channels: int
    num_heads: int = 1
    num_head_channels: int = -1
    use_new_attention_order: bool = False

    @nn.compact
    def __call__(self, x):
        # Compute num_heads
        if self.num_head_channels == -1:
            num_heads = self.num_heads
        else:
            num_heads = self.channels // self.num_head_channels

        B, *spatial, C = x.shape  # get dimensions
        # flatten x
        x = jnp.reshape(x, (B, -1, C))
        # normalize
        h = normalization(C)(x)
        # convolve to get qkv
        qkv = nn.Conv(features=num_heads * self.channels * 3,  # Getting the qkv tensor for multi-head attention
                      kernel_size=1)(h)
        # attention
        h = QKVAttention(num_heads)(qkv)
        # transpose to (B, d_model, C)
        h = h.transpose((0, 2, 1))  # getting channels last for convolutions
        # projection layer
        h = (nn.Conv(features=self.channels, kernel_size=1))(h)  # convolutional layer to get the value channels back to
        # the original size. Thus, things do not explode with increased number of heads.
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
        return jax.vmap(QKVAttention._forward)(q, k, v)

    @staticmethod
    def _forward(q, k, v) -> jnp.ndarray:
        """
        Args:
            q: the query tensor (H, T, d_model)
            k: the key tensor (H, T, d_model)
            v: the value tensor (H, T, d_model)
        where H is the number of heads, T is the number of elements that attend to each other
        and C is d_model
        Returns: a value tensor of shape (T * H, d_model) after attention. This tensor will be
        convolved with a 1x1 convolutional layer to recover the original shape (T, d_model) upstream.

        This function does not take into account the batch dimension. It will be vmap
        transformed to do so.
        (unit-tested)
        """
        headed_v = jax.vmap(QKVAttention.scaled_dot_product_attention)(q, k, v)  # (H, T, d_model/num_heads)
        # Merge the heads to recover v with dims (T, d_model).
        v_prime = jnp.concatenate(headed_v, axis=0)  # (T * num_heads, d_model)
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
