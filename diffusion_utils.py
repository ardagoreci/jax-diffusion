"""
Utility module
Defines the functions and classes required by to implement the diffusion model
"""

import jax
import jax.numpy as jnp

T = 1000  # Number of time steps for the diffusion process
DIFFUSION_CONSTANTS = jnp.linspace(start=10**(-4),  # defines the diffusion constants as specified in (Ho et al., 2020)
                                   stop=0.02,
                                   num=T)
alpha_array = 1 - DIFFUSION_CONSTANTS  # computes alphas
log_alphas = jnp.log(alpha_array)  # computes the log for numerical stability
MEAN_ALPHA_T = jnp.exp(jnp.cumsum(log_alphas))  # computes the mean of the alpha_ts for a given timestep


def diffuse_image(image, timestep, epsilon) -> jnp.ndarray:
    """Computes the corrupted image at a given timestep given the image,
    the timestep, and the epsilon.
    Args:
      params: The model parameters.
      image: The input image.
      timestep: the timestep for the diffusion Markov Chain
      epsilon: the sampled noise from a Gaussian of mean 0 and identity covariance.
    Returns: the epsilon prediction of the model

    1. Compute mean_alpha_t = alpha_t_1*alpha_t_2*...*alpha_t_T
    2. Compute x_t = image*sqrt(mean_alpha_t) + sqrt(1-mean_alpha_t)*epsilon
    3. Return x_t
    Returns: the noised image at the given timestep

    Note: this function is written for a single example. It is assumed
    that the pmap transformation will be used to apply it to the entire batch.
    """
    mean_alpha_t = MEAN_ALPHA_T[timestep]
    x_t = image*jnp.sqrt(mean_alpha_t) + jnp.sqrt(1-mean_alpha_t)*epsilon  # computes x_t
    return x_t


def sample(key, params, model, initial_noise):
    """
    Samples from the diffusion model with the iterative denoising procedure.
    Args:
        key: the random key for the jax.random.PRNGKey
        params: the model parameters
        model: the model with an apply function
        initial_noise: a batch of noise to start the diffusion process
    Returns: the sampled image
    """
    batch_size, *_ = initial_noise.shape
    x_t = initial_noise
    for t in range(T, 1, -1):
        if t > 1:
            key = jax.random.split(key, 1)[0] # get new key
            z = jax.random.normal(key, shape=initial_noise.shape) # sample z from Gaussian
        else:
            z = 0
        epsilon_theta = model.apply(params, x_t, jnp.repeat(t, batch_size))
        alpha_t = 1-DIFFUSION_CONSTANTS[t]
        mean_alpha_t = MEAN_ALPHA_T[t]
        # Denoise according to the formula in the paper
        sigma_t = jnp.sqrt(DIFFUSION_CONSTANTS[t])
        x_t = 1/alpha_t*(x_t - (1-alpha_t)/(jnp.sqrt(1-mean_alpha_t))*epsilon_theta) + z*sigma_t
    return x_t


