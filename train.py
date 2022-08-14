# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import ml_collections


def create_model():
    pass


def initialized(key, image_size, model):
    pass


def cross_entropy_loss(logits, labels):
    pass


def compute_metrics(logits, labels):
    pass


def create_learning_rate_fn(
        config: ml_collections.ConfigDict,
        base_learning_rate: float,
        steps_per_epoch: int):
    pass


def train_step(state, batch, learning_rate_fn):
    pass


def eval_step(state, batch):
    pass


def prepare_tf_data(xs):
    # TODO: this function should be implemented within data.py, not here.
    pass


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
    pass


def restore_checkpoint(state, workdir):
    pass


def save_checkpoint(state, workdir):
    pass


def create_train_state(rng,
                       config: ml_collections.ConfigDict,
                       model,
                       image_size,
                       learning_rate_fn):
    pass


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
    # rng = jax.random.PRNGKey(0)
    # compute local_batch_size (with the appropriate divisibility assertion)
    # compute num_train_steps
    # steps_per_checkpoint = steps_per_epoch * 10
    # base_learning_rate = config.learning_rate * config.batch_size / 256.

    # Create model
    # create learning rate function
    # create train_state
    # restore checkpoint
    # pmap transform train_step and eval_step
    #
    pass

