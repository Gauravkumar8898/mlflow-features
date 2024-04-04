import mlflow
import tensorflow as tf
from mlflow import MlflowClient
from mlflow.data.numpy_dataset import NumpyDataset
from mlflow.models import infer_signature

from src.utils.constant import batch_size, validation_split, verbose, epochs, artifact_location, experiment_tags


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
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
                  verbose=verbose)
        return model

    @staticmethod
    def runner_for_mnist():

        obj = MnistNeuralNetwork()

        client = MlflowClient()

        experiment = mlflow.set_experiment(experiment_name="Mnist_Model")
        if experiment:
            pass
        else:
            # Create the Experiment, providing a unique name
            experiment = client.create_experiment(
                name="Mnist_Model",
                artifact_location="mlflow_artifact",
                tags=experiment_tags
            )

        # autologging
        mlflow.autolog()

        description = "Integrate mlflow with Tensorflow and explore the full range of available features."
        with mlflow.start_run(run_name="tensorflow-mlflow", description=description,
                              experiment_id=experiment.experiment_id) as run:
            x_train, y_train, x_test, y_test = obj.load_mnist_dataset()
            # define dataset tracking
            train: NumpyDataset = mlflow.data.from_numpy(x_train, targets=y_train)
            test: NumpyDataset = mlflow.data.from_numpy(x_test, targets=y_test)
            mlflow.log_input(train, context="train")
            mlflow.log_input(test, context="test")
            model = obj.create_model()
            # Log the parameters
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "validation_split": validation_split,
                    "verbose": verbose
                }
            )

            model = obj.model_fit(x_train, y_train, model)
            test_loss, test_acc = model.evaluate(x_test, y_test)
            mlflow.log_metrics(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                }
            )

            #

            predictions = model.predict(x_test)
            signature = infer_signature(x_test[0], predictions[0])
            # #

            mlflow.tensorflow.log_model(model, artifact_location, signature=signature)

            # model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)

            model_name = "model"
            model_uri = mlflow.get_artifact_uri("model")
            if model_uri:
                pass
            else:
                # registered model
                # type 1
                # model_details = mlflow.register_model(model_uri=model_uri, name="model")
                # type 2

                # create registered model
                client.create_registered_model(model_name)

            # create model version
            # source = ""
            # run_id = "da1d5bd925d94977af9247904b43cacd"
            client.create_model_version(name=model_name, source=run.info.artifact_uri, run_id=run.info.run_id)

            # transition model version stage
            client.transition_model_version_stage(name=model_name, version="1", stage="Staging")
            client.transition_model_version_stage(name=model_name, version="2", stage="Archived")

            # delete model version
            # client.delete_model_version(name=model_name, version="1")

            # delete registered model
            # client.delete_registered_model(name=model_name)

            # adding description to registered model.
            client.update_registered_model(name=model_name, description="This is a test model")

            # adding tags to registered model.
            client.set_registered_model_tag(name=model_name, key="tag1", value="value1")

            # adding description to model version.
            client.update_model_version(name=model_name, version="1", description="This is a test model version")

            # adding tags to model version.
            client.set_model_version_tag(name=model_name, version="1", key="tag1", value="value1")

        mlflow.end_run()
