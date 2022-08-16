# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import functools
import time

import flax
import ml_collections

import jax
import optax
from jax import lax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from flax.training.train_state import TrainState
from flax.training import checkpoints
from flax.training import common_utils
from clu import metric_writers
from clu import periodic_actions
from absl import logging

import input_pipeline
import unet


def create_unet(config):
    """Creates and initializes the UNet model."""
    model = unet.UNetModel(in_channels=config.in_channels,
                           out_channels=config.out_channels,
                           num_res_blocks=config.num_res_blocks,
                           model_channels=config.model_channels,
                           attention_resolutions=config.attention_resolutions,
                           channel_mult=config.channel_mult)
    return model


def initialize(key, image_size, model, local_batch_size):
    """Utility function to initialize the model."""
    dummy_x = jnp.zeros((local_batch_size, image_size, image_size, 1))
    dummy_timesteps = jnp.zeros((local_batch_size,))
    params = model.init(key, dummy_x, dummy_timesteps)
    return params


def mean_squared_error(logits, labels):
    """Computes the element-wise mean squared error between logits and labels."""
    return jnp.mean(jnp.square(logits - labels))


def compute_metrics(logits, labels):
    """Returns a dictionary of metrics for the given logits and labels."""
    mse = mean_squared_error(logits, labels)
    return {'loss': mse}


def create_learning_rate_fn(config: ml_collections.ConfigDict):
    # TODO: implement the schedule that the authors have used.
    def _base_fn(step):
        return config.learning_rate

    return _base_fn


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def train_step(state: TrainState,
               batch,
               timesteps) -> TrainState:
    """Perform a single training step."""

    def loss_fn(params):
        logits = state.apply_fn(params, batch.images, timesteps)
        loss = mean_squared_error(logits, batch.labels)
        return loss, logits

    # Compute gradient
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (logits, aux), grads = grad_fn(state.params)
    # Update parameters (all-reduce gradients)
    grads = jax.lax.pmean(grads)
    metrics = compute_metrics(logits, batch.labels)

    # Update train state
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


@jax.jit
def eval_step(state, batch, timesteps):
    """Perform a single evaluation step."""
    logits = state.apply_fn(state.params, batch.images, timesteps)
    return compute_metrics(logits, batch.labels)


def create_input_iter(name: str,
                      split: str,
                      batch_size: int,
                      image_size: int,
                      cache: bool,
                      dtype):
    """Creates an iterator for the given dataset and split.
    Args:
        name: name of the dataset (specified in the config file)
        split: split of the dataset ('train', 'test')
        batch_size: batch size
        image_size: an integer specifying the size of the input images (not arbitrary
                    given UNet architecture)
        cache: whether to cache the dataset in memory
    Returns:
        an iterator of Batch named tuples containing the image and labels.

    TODO: this function does not return a dataset for the diffusion task!
    It is meant to be a test function for the rest.
    """
    dataset = input_pipeline.create_split(name=name, split=split,
                                          batch_size=batch_size, cache=cache)
    dataset = input_pipeline.preprocess_image_dataset(dataset, image_size, dtype=dtype)
    dataset = input_pipeline.make_denoising_dataset(dataset)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    iterator = map(prepare_tf_data, dataset)
    # iterator = input_pipeline.convert2iterator(dataset)
    # TODO: there is an issue with the prefetch.
    iterator = flax.jax_utils.prefetch_to_device(iterator, size=2)
    return iterator


def save_checkpoint(workdir, state):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, target=state, step=step, keep=3)


def create_train_state(rng,
                       config: ml_collections.ConfigDict,
                       model,
                       learning_rate_fn):
    """
    Creates the initial train state object.
    Args:
        rng: random number generator
        config: hyperparameter configuration
        model: Flax model
        image_size: integer specifying the height and width of input images
        learning_rate_fn: function that returns the learning rate for a given step.
    Returns:
        the initial train state object.
    """
    params = initialize(rng, config.image_size, model, local_batch_size=128)
    optimizer = optax.adam(learning_rate_fn)
    opt_state = optimizer.init(params)
    state = TrainState(apply_fn=model.apply,
                       params=params,
                       tx=optimizer,
                       step=0,
                       opt_state=opt_state)
    return state


def summarize_metrics(metrics):
    """Summarizes the metrics."""
    # TODO: this method might be choking the training loop
    summary = {}
    for metric in metrics:
        for key, value in metric.items():
            if summary.get(key) is None:
                summary[key] = value
            else:
                summary[key] += value
    # Average metrics
    for key, value in summary.items():
        summary[key] = value / len(metrics)
    return summary


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str):
    """
    Executes model training and evaluation loop.
    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the Tensorboard summaries are written to.

    Returns:
        final train state.
    """
    # Initialize writer
    writer = metric_writers.create_default_writer(logdir=workdir,
                                                  just_logging=jax.process_index() != 0)  # TODO: what??
    rng = jax.random.PRNGKey(0)
    image_size = config.image_size
    # compute local_batch_size (with the appropriate divisibility assertion)
    if config.batch_size % jax.device_count() > 0:
        raise ValueError("Global batch size should be divisible by the number of devices.")
    local_batch_size = config.batch_size // jax.process_count()  # TODO: what is the difference between process_count
    # and device_count?
    print(f"Local batch size: {local_batch_size}")
    platform = jax.local_devices()[0].platform

    if config.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    # Create input iterators
    train_iter = create_input_iter(name=config.dataset,
                                   split='train',
                                   batch_size=local_batch_size,
                                   image_size=image_size,
                                   cache=config.cache,
                                   dtype=input_dtype)
    test_iter = create_input_iter(name=config.dataset,
                                  split='test',
                                  batch_size=local_batch_size,
                                  image_size=image_size,
                                  cache=config.cache,
                                  dtype=input_dtype)
    # Compute num_train_steps
    steps_per_epoch = config.steps_per_epoch
    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps
    steps_per_checkpoint = config.steps_per_checkpoint

    if config.steps_per_eval == -1:
        steps_per_eval = 1000  # TODO: this is hard-coded for now.
    else:
        steps_per_eval = config.steps_per_eval

    # Create model
    model = create_unet(config)
    # Create learning rate function
    learning_rate_fn = create_learning_rate_fn(config)
    # Create train_state
    state = create_train_state(rng, config, model, learning_rate_fn)
    # restore checkpoint
    state = checkpoints.restore_checkpoint(workdir, state)
    # step_offset > 0 if we are resuming training
    step_offset = int(state.step)  # 0 usually
    state = flax.jax_utils.replicate(state)

    # pmap transform train_step and eval_step
    p_train_step = jax.pmap(
        functools.partial(train_step, learning_rate_fn=learning_rate_fn),
        axis_name='batch'
    )
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    # Create train loop
    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info("Initial compilation, this might take some minutes...")
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        # repeat_timesteps = jnp.repeat(jnp.arange(0, local_batch_size),  # TODO: temporary workaround
        #                              repeats=jax.device_count(), axis=0)
        repeat_timesteps = flax.jax_utils.replicate(jnp.arange(0, local_batch_size))
        state, metrics = p_train_step(state, batch, repeat_timesteps)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info("Initial compilation done.")
        if config.log_every_n_steps:
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_n_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)  # TODO: this is problematic with single device!
                summary = {
                    f'train_{k}': v
                    for k, v in jax.tree_util.tree_map(lambda x: x.mean(), train_metrics).items()
                }
                # summary = summarize_metrics(train_metrics)
                summary['steps_per_second'] = config.log_every_n_steps / (
                        time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

            if (step + 1) % steps_per_epoch == 0:
                epoch = step // steps_per_epoch
                eval_metrics = []
                for _ in range(steps_per_eval):
                    eval_batch = next(test_iter)
                    # repeat_timesteps = jnp.repeat(jnp.arange(0, local_batch_size),  # TODO: temporary workaround
                    #                              repeats=jax.device_count(), axis=0)
                    repeat_timesteps = flax.jax_utils.replicate(jnp.arange(0, local_batch_size))
                    metrics = p_eval_step(state, eval_batch, repeat_timesteps)
                    eval_metrics.append(metrics)
                eval_metrics = common_utils.get_metrics(eval_metrics)
                summary = jax.tree_util.tree_map(lambda x: x.mean(), eval_metrics)
                # summary = summarize_metrics(eval_metrics)
                logging.info('eval epoch: %d, loss: %.4f',
                             epoch, summary['loss'])
                writer.write_scalars(
                    step + 1, {f'eval_{key}': val for key, val in summary.items()})
                writer.flush()
            if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
                save_checkpoint(workdir, state)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
    return state
