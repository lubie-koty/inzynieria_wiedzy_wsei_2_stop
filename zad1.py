from collections import namedtuple
from numbers import Number

import numpy as np

from boston_data import prepare_boston_data

ModelTuple = namedtuple(
    'ModelTuple',
    ['W1', 'b1', 'W2', 'b2']
)


def sigmoid(number: Number) -> Number:
    return 1.0 / (1 + np.exp(-number))


def sigmoid_derivative(number: Number) -> Number:
    temp = sigmoid(number)
    return temp * (1.0 - temp)


def train(X, y, n_hidden, learning_rate, n_iter) -> ModelTuple:
    m, n_input = X.shape
    W1 = np.random.randn(n_input, n_hidden)
    b1 = np.zeros((1, n_hidden))
    W2 = np.random.randn(n_hidden, 1)
    b2 = np.zeros((1, 1))

    for i in range(1, n_iter + 1):
        Z2 = np.matmul(X, W1) + b1
        A2 = sigmoid(Z2)
        Z3 = np.matmul(A2, W2) + b2
        A3 = Z3
        dZ3 = A3 - y
        dW2 = np.matmul(A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0, keepdims=True)
        dZ2 = np.matmul(dZ3, W2.T) * sigmoid_derivative(Z2)
        dW1 = np.matmul(X.T, dZ2)
        db1 = np.sum(dZ2, axis=0)
        W2 = W2 - learning_rate * dW2 / m
        b2 = b2 - learning_rate * db2 / m
        W1 = W1 - learning_rate * dW1 / m
        b1 = b1 - learning_rate * db1 / m

        if i % 100 == 0:
            cost = np.mean((y - A3) ** 2)
            print(f'Iteracja {i}, strata: {cost}')
    
    return ModelTuple(W1, b1, W2, b2)


def predict(X_test, model: ModelTuple):
    Z2 = np.matmul(X_test, model.W1) + model.b1
    return np.matmul(sigmoid(Z2), model.W2) + model.b2


if __name__ == '__main__':
    n_hidden = 20
    learning_rate = 0.1
    n_iter = 2000

    X_train, X_test, y_train, y_test = prepare_boston_data()

    model = train(X_train, y_train, n_hidden, learning_rate, n_iter)
    predictions = predict(X_test, model)

    print(f'{predictions=}')
    print(f'{y_test=}')
