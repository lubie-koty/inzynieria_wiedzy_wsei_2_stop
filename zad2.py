import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from gen_features import generate_features


if __name__ == '__main__':
    data_raw = pd.read_csv('Corrected_Akcje_dane_8.csv', index_col='Date')
    data = generate_features(data_raw)

    start_train = '1992-01-02'
    end_train = '2022-12-30'
    start_test = '2023-01-03'
    end_test = '2023-12-15'

    data_train = data.loc[start_train:end_train]
    X_train = data_train.drop('close', axis=1).to_numpy()
    y_train = data_train['close'].to_numpy()
    
    data_test = data.loc[start_test:end_test]
    X_test = data_test.drop('close', axis=1).to_numpy()
    y_test = data_test['close'].to_numpy()

    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)

    tf.random.set_seed(42)
    model = Sequential()
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1))
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(0.21)
    )
    model.fit(
        X_scaled_train,
        y_train,
        epochs=1000,
        verbose=True
    )

    predictions = model.predict(X_scaled_test)[:, 0]
    print(f'Blad srednio-kwadratowy: {mean_squared_error(y_test, predictions):.3f}')
    print(f'Blad srednio-bezwarunkowy: {mean_absolute_error(y_test, predictions):.3f}')
    print(f'R^2: {r2_score(y_test, predictions):.3f}')

    plt.plot(data_test.index, y_test, c='k')
    plt.plot(data_test.index, predictions, c='b')
    plt.plot(data_test.index, predictions, c='r')
    plt.plot(data_test.index, predictions, c='g')
    plt.xticks(range(0, 252, 10), rotation=60)
    plt.xlabel('Data')
    plt.ylabel('Cena zamkniecia')
    plt.legend(['Wartosci rzeczywiste', 'Prognozy sieci neuronowej'])
    plt.show()
