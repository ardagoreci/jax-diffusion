"""Default hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()
    config.model = 'UNetModel'
    config.dataset = 'celeb_a'

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

    # Model hyperparameters
    config.model_channels = 4
    config.in_channels = 3
    config.out_channels = 3
    config.num_res_blocks = 3
    config.attention_resolutions = ()
    config.channel_mult = (1, 2, 4, 8)
    config.num_heads = 1
    config.num_head_channels = -1

    return config
