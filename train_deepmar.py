import os
import os

import tensorflow as tf

from lepar.datasets import PETAGenerator
from lepar.models import DeepMAR


def _map_fn(x, y):
    x = tf.keras.applications.resnet.preprocess_input(x)
    return x, y


if __name__ == "__main__":
    if not os.path.exists('out'):
        os.makedirs('out')

    # Define dataset and apply preprocessing steps
    dataset_parser = PETAGenerator()
    dataset = tf.data.Dataset.from_generator(
        dataset_parser.parse, output_signature=dataset_parser.output_signature
    )
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(64)

    # Define model
    model = DeepMAR(35)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=tf.keras.metrics.BinaryCrossentropy(from_logits=True),
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="out/deepmar/logs", histogram_freq=1
    )

    # Fit model on dataset
    model.fit(dataset, epochs=100, callbacks=[tensorboard_callback])

    # Save model when done training
    model.save_weights('out/deepmar')