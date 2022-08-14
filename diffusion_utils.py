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


def _forward_fn(self, params, image, timestep, epsilon) -> jnp.ndarray:
    """The forward function of the model.
    Args:
      params: The model parameters.
      image: The input image.
      timestep: the timestep for the diffusion Markov Chain
      epsilon: the sampled noise from a Gaussian of mean 0 and identity covariance.
    Returns: the epsilon prediction of the model

    1. Compute mean_alpha_t = alpha_t_1*alpha_t_2*...*alpha_t_T
    2. Compute x_t = image*sqrt(mean_alpha_t) + sqrt(1-mean_alpha_t)*epsilon
    3. Positional encode x_t with the timestep
    3. Feed to the model, get epsilon prediction

    Note: this function is written for a single example. It is assumed
    that the pmap transformation will be used to apply it to the entire batch.
    """
    alpha_array = 1-DIFFUSION_CONSTANTS  # computes alphas
    log_alphas = jnp.log(alpha_array)  # computes the log for numerical stability
    mean_alpha_t = jnp.exp(jnp.sum(log_alphas[:timestep+1]))  # sum is equivalent to multiplying under log, then revert
    x_t = image*jnp.sqrt(mean_alpha_t) + jnp.sqrt(1-mean_alpha_t)*epsilon  # computes x_t
    # Positional encode x_t with the timestep according to Transformers encoding scheme


def loss_fn(params, key, batch) -> jnp.ndarray:
    """Computes the diffusion loss for a batch of images.

    Args:
        params: A dictionary of parameters.
        key: A random key for jax.random.split.
        batch: A batch of images.

    Returns:
        a jnp.ndarray of shape (batch_size,) containing the loss for each image in the batch.
    1. Sample t from uniform({1, ..., T})
    2. Sample epsilon from Gaussian(0, I)
    (assert epsilon.shape == image.shape)
    3. Compute epsilon_theta = _forward_fn(params, image, t, epsilon)
    4. Compute mean squared error of (epsilon - epsilon_theta)**2
    5. Sum all the elements in (epsilon - epsilon_theta)**2
    """
    batch_size, *_ = batch.images.shape
    # TODO: find a more elegant way of computing latent_dim
    latent_dim = batch.shape[1]*batch.shape[2]*batch.shape[3]  # latent_dim = 784
    epsilon_batch = jax.random.multivariate_normal(key,
                                                   mean=jnp.zeros(latent_dim),
                                                   cov=jnp.identity(latent_dim),
                                                   shape=batch_size)
    epsilon_batch = jnp.reshape(epsilon_batch, batch.images.shape)
    timestep = jnp.random.randint(key, 1, T)
    epsilon_theta_batch = _forward_fn(params, batch.images, timestep, epsilon_batch)
    return 0.0
