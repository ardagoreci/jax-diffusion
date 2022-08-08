"""
Data pipeline for the diffusion model.
"""
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from typing import NamedTuple, Iterator
from tensorflow.keras.applications import resnet_v2


class Batch(NamedTuple):
    images: np.ndarray
    labels: np.ndarray


def load_dataset(name: str,
                 split: str,
                 shuffle: bool = False,
                 batch_size: int = 128) -> Iterator:
    """Loads the dataset.
    Args:
      name: Name of the dataset.
      split: The split of the dataset to load.
      shuffle: Whether to shuffle the dataset.
      batch_size: The batch size.

    Returns:
      An iterator over the dataset.

    >>> dataset = load_dataset(name='mnist', split='train')
    >>> print(next(dataset).images.shape)
    (128, 28, 28, 1)
    """
    ds = tfds.load(name,
                   split=split,
                   shuffle_files=shuffle,
                   as_supervised=True).cache().repeat()

    def normalize(images, labels):
        images = tf.cast(images, dtype='float32')
        images = resnet_v2.preprocess_input(images)
        return images, labels

    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.map(normalize)
    ds = ds.map(lambda x, y: Batch(x, y))
    return iter(tfds.as_numpy(ds))
