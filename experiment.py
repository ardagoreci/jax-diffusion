"""The module required for JAXline to log the training and evaluation.
TODO: there is a weird import error that I do not understand in this module.
I suspect it is related to Jaxline."""
import jaxline
from jaxline.experiment import AbstractExperiment
import jax
import haiku as hk
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Mapping, Optional, Text, Tuple, Iterator
from ml_collections import config_dict
import absl
import optax
from jaxline import platform
import jaxline.utils as jl_utils
from absl import flags
import sys
import data
import unet

FLAGS = flags.FLAGS
Scalars = Mapping[Text, jnp.ndarray]


def get_config():
    """Return config object for training"""
    config = jaxline.base_config.get_base_config()  # Get the base config and fine-tune it.

    # Dataset name
    config.dataset = 'mnist'

    # Experiment config.
    local_batch_size = 2
    # Modify this to adapt to your custom distributed learning setup
    num_devices = 1
    config.train_batch_size = local_batch_size * num_devices
    config.n_epochs = 110
    config.training_steps = 10000  # Number of training steps.

    # Intervals for logging.
    config.interval_type = "secs"
    config.save_checkpoint_interval = 300
    config.log_train_data_interval = 120.0  # None to turn off

    # If set to True checkpoint on all hosts, which may be useful
    # for model parallelism. Otherwise, checkpoint on host 0.
    config.train_checkpoint_all_hosts = False

    # If True, asynchronously logs training data from every training step.
    config.log_all_train_data = False

    # If true, run evaluate() on the experiment once before you load a checkpoint.
    # This is useful for getting initial values of metrics at random weights, or
    # when debugging locally if you do not have any train job running.
    config.eval_initial_weights = False

    # When True, the eval job immediately loads a checkpoint runs evaluate()
    # once, then terminates.
    config.one_off_evaluate = False

    # Number of checkpoints to keep by default
    config.max_checkpoints_to_keep = 5

    # Settings for the RNGs used during training and evaluation.
    config.random_seed = 42
    config.random_mode_train = "unique_host_unique_device"
    config.random_mode_eval = "same_host_same_device"

    # The metric (returned by the step function) used as a fitness score.
    # It saves a separate series of checkpoints corresponding to
    # those which produce a better fitness score than previously seen.
    # By default it is assumed that higher is better, but this behaviour can be
    # changed to lower is better, i.e. behaving as a loss score, by setting
    # `best_model_eval_metric_higher_is_better = False`.
    # If `best_model_eval_metric` is empty (the default), best checkpointing is
    # disabled.
    config.best_model_eval_metric = "epsilon_mse"
    config.best_model_eval_metric_higher_is_better = False
    config.checkpoint_dir = './checkpoints'

    # Prevents accidentally setting keys that aren't recognized.
    config.lock()
    return config


class Experiment(AbstractExperiment):
    """The class required for JAXline to log the training and evaluation."""

    def __init__(self, mode, init_rng, config):
        """
        Initialize the experiment.
        """
        super(Experiment, self).__init__(mode=mode, init_rng=init_rng)
        self.experiment_name = "Diffusion on MNIST"
        self.experiment_description = "Trains a diffusion model on MNIST, " \
                                      "visualizing the progress of the training."
        self.experiment_version = "1.0"
        self.config = config
        self.mode = mode
        self.init_rng = init_rng

        # Create the model
        self.mnist_model = hk.without_apply_rng(hk.transform(unet.MNISTClassifier))

        # Create the optimizer
        self.optimizer = optax.lamb(learning_rate=0.01)

        # Checkpointed experiment state
        self._params = self.model.init(init_rng)
        self._opt_state = self.optimizer.init(self._params)  # initialize optimizer with initial params

        # Input pipelines
        self._train_input = self._build_input_fn('train')
        self._eval_input = self._build_input_fn('test')

        self.forward = hk.transform(self._forward_fn)

        # self.update_fn = jax.pmap(self._update_fn)
        # self.eval_batch = jax.jit(self._eval_batch)

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
        pass

    def _loss_fn(self, params, key, batch) -> jnp.ndarray:
        """Computes the loss for a single training example.
        Args:
            params: the model parameters
            key: the key for the random number generator
            image: an image from the training set
        Returns: the loss for a single example

        1. Sample t from uniform({1, ..., T})
        2. Sample epsilon from Gaussian(0, I)
        (assert epsilon.shape == image.shape)
        3. Compute epsilon_theta = _forward_fn(params, image, t, epsilon)
        4. Compute mean squared error of (epsilon - epsilon_theta)**2
        5. Sum all the elements in (epsilon - epsilon_theta)**2

        For the MNIST implementation, it is batched
        replace batch with image
        """
        batch_size, *_ = batch.images.shape
        images = batch.images
        labels = batch.labels
        logits = self.mnist_model.apply(self._params, batch)

        labels = jax.nn.one_hot(labels, 10)  # One-hot encode with 10 classes
        log_likelihood = jnp.sum(labels * jax.nn.log_softmax(logits))

        return -log_likelihood / batch_size

    def _update_fn(
            self,
            params,
            key,
            batch
    ) -> Tuple[hk.Params, hk.OptState, Scalars]:
        """Learning rule for the model. Performs an update on the
        params and returns a new state.
        Args:
            params: the model parameters
            key: a random key
            batch: a batch of images from the training set

        Returns:
            params: the updated model parameters
            opt_state: updated optimizer state
            scalars: scalars to be logged
        1. Split the key to produce different random numbers for each example in the batch
        2. Compute per_example_grads = pmapped_grad_fn(params, key_batch, batch)
        3. All reduce gradients grads = jax.lax.pmean(per_example_grads)
        4. updates, opt_state = optimizer.update(grads, opt_state, params)
        5. params = optax.apply_updates(params, updates)
        return params, state, opt_state, scalars
        """
        key_batch = jax.random.split(key, self.config.batch_size)  # Split the key
        # TODO: implement steps 2 to 5 after the model is done
        grads = jax.grad(self._loss_fn)(params, key, batch)
        updates, opt_state = self.optimizer.update(grads, self._opt_state, params)

        params = optax.apply_updates(params, updates)
        scalars = {'global_gradient_norm': optax.global_norm(grads)}  # TODO: add the learning rate
        return params, opt_state, scalars

    def step(
            self,
            global_step: jnp.ndarray,
            rng: jnp.ndarray,
            writer: Optional[jaxline.utils.Writer],
    ) -> Dict[str, np.ndarray]:
        """Performs a step of computation e.g. a training step.
        This function will be wrapped by `utils.kwargs_only` meaning that when
        the user re-defines this function they can take only the arguments
        they want e.g. def step(self, global_step, **unused_args).
        Args:
          global_step: A `ShardedDeviceArray` of the global step, one copy
            for each local device. The values are guaranteed to be the same across
            all local devices, it is just passed this way for consistency with
            `rng`.
          rng: A `ShardedDeviceArray` of `PRNGKey`s, one for each local device,
            and unique to the global_step. The relationship between the keys is set
            by config.random_mode_train.
          writer: An optional writer for performing additional logging (note that
            logging of the returned scalars is performed automatically by
            jaxline/train.py)
        Returns:
          A dictionary of scalar `np.array`s to be logged.
        """
        batch = next(self._train_input)  # (per_device_batch_size, H, W, C)
        params, state, opt_state, scalars = self._update_fn(self._params, rng, batch)
        # TODO: write the generated images via the writer to logs
        scalars = jl_utils.get_first(scalars)  # returns scalars from the first device
        return scalars

    def evaluate(
            self,
            global_step: jnp.ndarray,
            rng: jnp.ndarray,
            writer: Optional[jaxline.utils.Writer],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Performs the full evaluation of the model.
        This function will be wrapped by `utils.kwargs_only` meaning that when
        the user re-defines this function they can take only the arguments
        they want e.g. def evaluate(self, global_step, **unused_args).
        Args:
          global_step: A `ShardedDeviceArray` of the global step, one copy
            for each local device.
          rng: A `ShardedDeviceArray` of random keys, one for each local device,
            and, unlike in the step function, *independent* of the global step (i.e.
            the same array of keys is passed at every call to the function). The
            relationship between the keys is set by config.random_mode_eval.
          writer: An optional writer for performing additional logging (note that
            logging of the returned scalars is performed automatically by
            jaxline/train.py)
        Returns:
          A dictionary of scalar `np.array`s to be logged.

        logging.info ?
        scalars = jax.device_get(self._eval_epoch())
        """
        # TODO: implement a MSE metric for epsilon and a digit visualizer
        # TODO: return the MSE and write the digit via the writer
        pass

    def _eval_batch(self, params, key, batch) -> Scalars:
        """Evaluates the model on a batch of data.
        1. Split the key to produce different random numbers for each example in the batch
        2. per_example_epsilon_mse = pmapped_loss_fn(params, key_batch, batch)
        3. All reduce epsilon_mse = jax.lax.pmean(per_example_epsilon_mse)
        return epsilon_mse
        """
        pass

    def _eval_epoch(self, rng):
        """Evaluates an epoch.
        Args:
            rng: a random key
        Returns:
            scalars: scalars to be logged

        (I am not quite sure about the implementation here!)
        mean_scalars
        """
        pass

    def _build_input_fn(self, split) -> Iterator[data.Batch]:
        num_devices = jax.device_count()
        global_batch_size = self.config.batch_size
        per_device_batch_size, ragged = divmod(global_batch_size, num_devices)
        if ragged:
            raise ValueError(
                f"The batch size ({global_batch_size}) must be divisible by"
                f" the number of devices ({num_devices})"
            )
        return data.load_dataset(name=self.config.dataset,
                                 split=split,
                                 batch_size=per_device_batch_size)


if __name__ == '__main__':
    flags.mark_flag_as_required('config')
    platform.main(Experiment, sys.argv[1:])
