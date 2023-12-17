import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"
    n_hidden = 5
    n_iter = 2000
    learning_rate = 0.1
    random_state = 42

    raw_df = pd.read_csv(DATA_URL, sep="\s+", skiprows=22, header=None)
    X = np.hstack([raw_df.to_numpy()[::2, :], raw_df.to_numpy()[1::2, :2]])
    y = raw_df.to_numpy()[1::2, 2]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_params = {
        'activation': ['relu'],
        'batch_size': ['auto'],
        'hidden_layer_sizes': [n_hidden, n_hidden],
        'learning_rate': ['constant'],
        'learning_rate_init': [learning_rate],
        'max_iter': [n_iter],
        'solver': ['lbfgs']
    }

    rgr = GridSearchCV(MLPRegressor(), train_params, cv=5)
    rgr.fit(X_train, y_train)
    
    train_mse = mean_squared_error(y_train, rgr.predict(X_train))
    test_mse = mean_squared_error(y_test, rgr.predict(X_test))

    print(f'{rgr.best_score_=}')
    print(f'train mse: {np.round(train_mse, 2)}')
    print(f'test mse: {np.round(test_mse, 2)}')

