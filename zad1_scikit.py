import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from boston_data import prepare_boston_data


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = prepare_boston_data()

    mlp_regressor = MLPRegressor(
        hidden_layer_sizes=(16,8),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        random_state=42,
        max_iter=2000
    )
    mlp_regressor.fit(X_train, y_train)

    predictions = mlp_regressor.predict(X_test)
    test_mse = mean_squared_error(y_test, predictions)

    print(f'predictions: {predictions}')
    print(f'mean squared error: {np.round(test_mse, 2)}')
