from numbers import Number

import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs

        self.weights = np.array([[.50], [.50], [.50]])
        self.error_history = []
        self.epoch_list = []

    def sigmoid(self, x: Number, derivative: bool = False) -> Number:
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def feed_forward(self) -> None:
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def back_propagation(self) -> None:
        self.error = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, True)
        self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs: int = 25000) -> None:
        for epoch in range(epochs):
            self.feed_forward()
            self.back_propagation()
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)

    def predict(self, new_input) -> Number:
        return self.sigmoid(np.dot(new_input, self.weights))


if __name__ == '__main__':
    inputs = np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        [1, 0, 1]
    ])
    outputs = np.array([[0], [0], [0], [1], [1], [1]])

    neural_network = NeuralNetwork(inputs, outputs)
    neural_network.train()

    example_1 = np.array([[1, 1, 0]])
    example_2 = np.array([[0, 1, 1]])

    print(f'{neural_network.predict(example_1)} - Correct: {example_1[0][0]}')
    print(f'{neural_network.predict(example_2)} - Correct: {example_2[0][0]}')

    plt.figure(figsize=(15, 5))
    plt.plot(neural_network.epoch_list, neural_network.error_history)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
