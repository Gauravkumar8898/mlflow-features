import tensorflow as tf
from sklearn.model_selection import train_test_split
import mlflow
import pandas as pd
from src.utils.constant import experiment_tags
from mlflow import MlflowClient
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import os
import time

#
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"


class IrisNeuralNetwork:

    @staticmethod
    def iris_data_generator():
        """
        A function which generates the iris dataset
        """
        iris = load_iris()
        iris_dataset = pd.DataFrame(
            data=iris.data, columns=iris.feature_names
        )
        iris_labels = pd.DataFrame(data=iris.target, columns=["label"])
        iris_dataset = pd.concat(
            [iris_dataset, iris_labels], axis=1
        )
        return iris_dataset, "label"


    @staticmethod
    def train(dataset, label_column: str = "label", ):
        # Get the input dataframe (Use DataItem.as_df() to access any data source)
        df = dataset
        print(df)
        print(type(df))

        # Initialize the x & y data
        X = df.drop(label_column, axis=1)
        y = df[label_column]

        # Train/Test split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pick an ideal ML modelataset,

        # Create a sequential model
        model = tf.keras.Sequential([
            # First dense layer with 23 units and ReLU activation function
            tf.keras.layers.Dense(23, activation='relu'),
            # Dropout layer with a dropout rate of 20%
            tf.keras.layers.Dropout(0.2),
            # Second dense layer with 15 units and ReLU activation function
            tf.keras.layers.Dense(15, activation='relu'),
            # Batch normalization layer
            tf.keras.layers.BatchNormalization(),
            # Third dense layer with 10 units and linear activation function
            tf.keras.layers.Dense(10, activation='linear'),
            # Output layer with 1 unit and sigmoid activation function
            tf.keras.layers.Dense(1, activation='sigmoid')])

        # compile the model
        model.compile(loss=tf.keras.losses.mae,
                      optimizer=tf.keras.optimizers.Adam(),
                      metrics=['accuracy'])

        return model, X_train, X_test, y_train, y_test
        # Train the model

    @staticmethod
    def runner_for_iris():
        # autologging
        mlflow.autolog()
        obj = IrisNeuralNetwork()

        client = MlflowClient()

        experiment = mlflow.set_experiment(experiment_name="Iris_Model")
        if experiment:
            pass
        else:
            # Create the Experiment, providing a unique name
            experiment = client.create_experiment(
                name="Iris_Model",
                artifact_location="mlflow_artifact",
                tags=experiment_tags
            )

        description = "Integrate mlflow with Tensorflow and explore the full range of available features."
        with mlflow.start_run(run_name="tensorflow-mlflow", description=description,
                              log_system_metrics=True, experiment_id=experiment.experiment_id) as run:
            time.sleep(15)
            dataset, label = obj.iris_data_generator()
            model, X_train, X_test, y_train, y_test = obj.train(dataset, label)
            dataset = mlflow.data.from_pandas(dataset,targets=label)
            mlflow.log_input(dataset,context="dataset")
            model.fit(X_train, y_train, epochs=10, validation_split=0.2)

            model_name = "Iris model"
            model_uri = mlflow.get_artifact_uri("model")
            # if model_uri:
            #     pass
            # else:
                # registered model
                # type 1
            # model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
                # type 2

                # create registered model
                # client.create_registered_model(model_name)

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

        print(mlflow.MlflowClient().get_run(run.info.run_id).data)
