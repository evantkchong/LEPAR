import os

import tensorflow as tf

from lepar.datasets import PETAGenerator
from lepar.models import LEPAR
from lepar.loss import MultiLabelTripletSemiHard


def _map_fn(x, y):
    x = tf.keras.applications.resnet.preprocess_input(x)
    return x, y


if __name__ == "__main__":
    # Define dataset and apply preprocessing steps
    batch_size = 64
    dataset_parser = PETAGenerator()
    dataset = tf.data.Dataset.from_generator(
        dataset_parser.parse, output_signature=dataset_parser.output_signature
    )
    dataset = dataset.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)

    # Define model
    model = LEPAR(256)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(), loss=MultiLabelTripletSemiHard()
    )

    # Fit model on dataset
    model.fit(dataset, epochs=2)

    # Save model when done training
    if not os.path.exists("out"):
        os.makedirs("out")
    model.save_weights("out/lepar")
