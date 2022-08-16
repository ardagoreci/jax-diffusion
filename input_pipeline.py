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
    (deprecated)

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


def create_split(name: str,
                 split: str,
                 shuffle: bool = False,
                 batch_size: int = 128,
                 cache: bool = False) -> tf.data.Dataset:
    """Creates an iterator over the dataset.
    Args:
      name: Name of the dataset.
      split: The split of the dataset to load.
      shuffle: Whether to shuffle the dataset.
      batch_size: The batch size.
      cache: Whether to cache the dataset.

    Returns:
      A batched dataset.
    """
    ds = tfds.load(name,
                   split=split,
                   shuffle_files=shuffle,
                   as_supervised=True)
    if cache:
        ds = ds.cache()
    ds = ds.repeat()
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def preprocess_image_dataset(dataset, image_size: int,
                             dtype: str = 'float32') -> tf.data.Dataset:
    """
    Normalizes images with the ResNet preprocessing function to [-1, 1]
    and resizes them to the specified size.
    """
    def resize(images, labels):
        images = tf.image.resize(images, [image_size, image_size])
        return images, labels

    def normalize(images, labels):
        images = tf.cast(images, dtype=dtype)
        images = resnet_v2.preprocess_input(images)
        return images, labels

    dataset = dataset.map(resize)
    dataset = dataset.map(normalize)
    return dataset


def make_denoising_dataset(dataset):
    def corrupt(images, labels):
        noise = tf.random.normal(shape=images.shape, mean=0.0, stddev=1.0)
        noised = images + noise
        return noised, images
    dataset = dataset.map(corrupt)
    return dataset


def convert2iterator(ds):
    ds = ds.map(lambda x, y: Batch(x, y))
    return iter(tfds.as_numpy(ds))


def prepare_tf_data(xs):
    # TODO: implement this function
    pass
