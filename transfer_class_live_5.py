import tensorflow as tf

# All the data loading stuff is the same as in the function approach.
#   So we will skip that.
# So, I'm assuming that we have train and validation datasets at this point.

class Model:
    def __init__(self, input_shape):
        self.model = tf.keras.Sequential()
        self.res = tf.keras.applications.resnet50.ResNet50(
            include_top = False,
            input_shape = input_shape,
            pooling = "avg",
        )
        self.res.trainable = False
        self.model.add(self.res)
        # Only need to Flatten if pooling = None
        self.model.add(tf.keras.layers.Flatten())
        # The positional argument = # of categories in your dataset
        # Softmax re-scales your predictions to be betw 0 & 1, adding up to 1.
        self.model.add(tf.keras.layers.Dense(5, activation = "softmax"))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        self.loss = tf.keras.losses.CategoricalCrossentropy()

        self.model.compile(
            optimizer = self.optimizer,
            loss = self.loss,
            metrics = ["accuracy"], # Affects the printout only
        )

model = Model((224, 224, 3))

model.model.summary()

# Slight difference here: first model is an instance, second model is attribute
# model.model.fit()