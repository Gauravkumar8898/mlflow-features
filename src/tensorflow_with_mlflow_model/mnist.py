import tensorflow as tf
import mlflow


class MnistNeuralNetwork:

    @staticmethod
    def load_mnist_dataset():
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    @staticmethod
    def create_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model

    @staticmethod
    def model_fit(x_train, y_train, model):
        model.fit(x_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=2)
        return model

    @staticmethod
    def runner_for_mnist():
        mlflow.autolog()
        obj = MnistNeuralNetwork()
        x_train, y_train, x_test, y_test = obj.load_mnist_dataset()
        model = obj.create_model()
        model = obj.model_fit(x_train, y_train, model)
        print(model.evaluate(x_test, y_test))



