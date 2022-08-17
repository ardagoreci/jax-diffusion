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


def create_split(name: str,
                 split: str,
                 data_dir: str,
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
      try_gcs: Whether to try to load the dataset from GCS. (for running on TPUs)
      data_dir: The directory to load the dataset from (GCS directory)

    Returns:
      A batched dataset.
    """
    # TODO: temporary workaround, find a more elegant solution
    if name == 'celeb_a':
        ds = tfds.load(name,
                       split=split,
                       shuffle_files=shuffle,
                       data_dir=data_dir)

        def map_fn(feature_dict):
            image = feature_dict['image']
            label = tf.ones((1,))
            return image, label

        ds = ds.map(map_fn)
    else:
        ds = tfds.load(name,
                       split=split,
                       shuffle_files=shuffle,
                       as_supervised=True,
                       data_dir=data_dir)

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
        images = tf.image.resize(images, [image_size, image_size],
                                 preserve_aspect_ratio=True)
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


def create_diffusion_dataset(name: str,
                             split: str,
                             image_size: int,
                             shuffle: bool = False,
                             batch_size: int = 128,
                             cache: bool = False,
                             data_dir=None,
                             dtype: str = 'float32'):
    """Creates the dataset for the diffusion model.
    """
    dataset = create_split(name=name,
                           split=split,
                           shuffle=shuffle,
                           batch_size=batch_size,
                           data_dir=data_dir,
                           cache=cache)
    dataset = preprocess_image_dataset(dataset, image_size, dtype)

    def _epsilon_label(images, labels):
        epsilon = tf.random.normal(shape=images.shape, mean=0.0, stddev=1.0)
        return images, epsilon

    dataset = dataset.map(_epsilon_label)
    return dataset


def convert2iterator(ds):
    ds = ds.map(lambda x, y: Batch(x, y))
    return iter(tfds.as_numpy(ds))


def batch_iterator(it):
    return map(lambda x: Batch(x[0], x[1]), it)


def prepare_tf_data(xs):
    # TODO: implement this function
    pass
