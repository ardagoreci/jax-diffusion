"""
Hyperparameter configuration file for MNIST.
MNIST will be used for testing the functionality of the framework.
"""
import ml_collections


def get_config():
    """Get the hyperparameter configuration for training on MNIST"""
    config = ml_collections.ConfigDict()
    config.model = 'UNetModel'
    config.dataset = 'mnist'

    config.learning_rate = 0.1
    config.batch_size = 256
    config.warmup_epochs = 5.0
    config.momentum = 0.9

    config.num_epochs = 100
    config.log_every_n_steps = 100

    config.cache = False
    config.half_precision = False

    config.num_train_steps = -1
    config.steps_per_eval = -1
    return config
