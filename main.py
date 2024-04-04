# from src.tensorflow_with_mlflow_model.neaural_network_mnist import MnistNeuralNetwork
# from src.tensorflow_with_mlflow_model import winequality_tensorflow
# from src.tensorflow_with_mlflow_model.mnist import MnistNeuralNetwork
from src.tensorflow_with_mlflow_model.mlflow_iris_classifier import IrisNeuralNetwork

if __name__ == '__main__':
    obj1 = IrisNeuralNetwork()
    obj1.runner_for_iris()
