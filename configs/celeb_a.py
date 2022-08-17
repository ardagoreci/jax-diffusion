"""
Hyperparameter configuration file for CelebA dataset.
"""
import default as default_config


def get_config():
    """Get the hyperparameter configuration for training on MNIST"""
    config = default_config.get_config()
    config.dataset = 'celeb_a'
    config.batch_size = 64
    config.cache = False
    config.data_dir = 'gs://celeb-a-diffusion'
    config.image_size = 128

    # Training hyperparameters
    config.steps_per_epoch = 160_000 // config.batch_size
    config.steps_per_eval = 20_000 // config.batch_size
    config.steps_per_checkpoint = 160_000 // config.batch_size  # save a checkpoint every epoch
    config.num_steps = 500_000  # for CelebA HQ, 500_000 steps were used
    # config.steps_per_eval = 100

    # Model hyperparameters
    config.learning_rate = 1e-4
    config.grad_clip = 0.012
    config.model_channels = 256
    config.in_channels = 1
    config.out_channels = 1
    config.num_res_blocks = 2
    config.num_heads = 2
    config.attention_resolutions = (16, 8)
    config.channel_mult = (1, 2, 4, 8)

    # Seed for reproducibility
    config.seed = 42

    return config