import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler


def prepare_boston_data(random_state=42):
    DATA_URL = 'http://lib.stat.cmu.edu/datasets/boston'
    NUM_TEST = 10

    raw_df = pd.read_csv(DATA_URL, sep='\s+', skiprows=22, header=None)
    data = np.hstack([raw_df.to_numpy()[::2, :], raw_df.to_numpy()[1::2, :2]])
    target = raw_df.to_numpy()[1::2, 2]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(data[:-NUM_TEST, :])
    y_train = target[:-NUM_TEST].reshape(-1, 1)
    X_test = scaler.transform(data[-NUM_TEST:, :])
    y_test = target[-NUM_TEST:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
