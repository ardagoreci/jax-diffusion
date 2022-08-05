"""
Defines the U-Net architecture that will be used for gradual denoising
of the latent space.
"""
import jax
import haiku as hk
import jax.numpy as jnp

