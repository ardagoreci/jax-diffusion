"""
Hyperparameter configuration file for MNIST.
MNIST will be used for testing the functionality of the framework.
"""
import default as default_config


def get_config():
    """Get the hyperparameter configuration for training on MNIST"""
    config = default_config.get_config()
    config.dataset = 'mnist'
    config.batch_size = 128

    # Model hyperparameters
    config.model_channels = 4
    config.in_channels = 1
    config.out_channels = 1
    config.num_res_blocks = 2
    config.attention_resolutions = ()
    config.channel_mult = (1, 2, 4, 8)

    return config
