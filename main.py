from src.tensorflow_with_mlflow_model.neaural_network_mnist import MnistNeuralNetwork


if __name__ == '__main__':
    obj = MnistNeuralNetwork()
    obj.runner_for_mnist()