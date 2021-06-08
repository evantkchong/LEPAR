import tensorflow as tf


class DeepMAR(tf.keras.Model):
    def __init__(self, num_classes, dropout=0.0):
        super(DeepMAR, self).__init__()
        self.feature_extractor = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", pooling="avg"
        )
        if dropout > 0.0:
            self.dropout = tf.keras.layers.Dropout(rate=dropout)
        else:
            self.dropout = None
        initializer = tf.keras.initializers.RandomNormal(stddev=0.001)
        self.head = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer=initializer,
        )

    def call(self, inputs, training=False):
        x = self.feature_extractor(inputs, training=training)
        if self.dropout and training:
            x = self.dropout(x, training=training)
        x = self.head(x, training=training)
        return x
