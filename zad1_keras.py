import numpy as np
import tensorflow as tf

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from boston_data import prepare_boston_data


if __name__ == '__main__':
    tf.random.set_seed(42)
    X_train, X_test, y_train, y_test = prepare_boston_data()

    model = Sequential([
        Dense(units=20, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=1, activation='relu')
    ])
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(0.02)
    )
    model.fit(X_train, y_train, epochs=300)

    predictions = model.predict(X_test)[:, 0]

    print(f'predictions: {predictions}')
    print(f'mean squared error: {np.mean((y_test - predictions) ** 2)}')
