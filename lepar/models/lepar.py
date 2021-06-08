import tensorflow as tf


class LEPAR(tf.keras.Model):
    """This LEPAR class instantiates the model we will train with the
    multi-label Triplet Loss. It is essentially just the resnet50 feature
    extractor from tf.keras.applications with an additional Dense Layer to
    generate embeddings with L2 Normalization applied.
    """

    def __init__(self, output_size, input_shape=None, dropout=0.5):
        super(LEPAR, self).__init__()
        self.feature_extractor = tf.keras.applications.ResNet50(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=input_shape,
        )
        self.flatten = tf.keras.layers.Flatten()
        if dropout > 0.0:
            self.dropout = tf.keras.layers.Dropout(rate=dropout)
        else:
            self.dropout = None
        self.dense = tf.keras.layers.Dense(output_size, activation=None)

    def call(self, inputs, training=False):
        x = self.feature_extractor(inputs, training=training)
        x = self.flatten(x)
        if self.dropout and training:
            x = self.dropout(x, training=training)
        x = self.dense(x, training=training)
        x = tf.math.l2_normalize(x, axis=1)
        return x
