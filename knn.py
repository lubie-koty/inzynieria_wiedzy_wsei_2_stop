import pandas

from collections import Counter
from numbers import Number
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def minkowski_distance(a: list[Number], b: list[Number], p: Number = 1) -> Number:
    dimension = len(a)
    distance = 0
    for i in range(dimension):
        distance += abs(a[i] - b[i]) ** p
    return distance ** (1 / p)


def knn_predict(X_train, X_test, y_train, k, p):
    y_hat_test = []
    for test_point in X_test:
        distances = [minkowski_distance(test_point, train_point, p) for train_point in X_train]
        df_dists = pandas.DataFrame(data=distances, columns=['dist'], index=y_train.index)
        df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]
        counter = Counter(y_train[df_nn.index])
        y_hat_test.append(counter.most_common()[0][0])
    return y_hat_test


if __name__ == '__main__':
    iris_dataset = datasets.load_iris()
    data_frame = pandas.DataFrame(data=iris_dataset.data, columns=iris_dataset.feature_names)
    data_frame['target'] = iris_dataset.target

    X = data_frame.drop('target', axis=1)
    y = data_frame.target
    print(f'minkowski_test={minkowski_distance(X.iloc[0], X.iloc[1], 1)}')

    test_pt = [4.8, 2.7, 2.5, 0.7]
    distances = [minkowski_distance(test_pt, X.iloc[i]) for i in X.index]

    df_dists = pandas.DataFrame(data=distances, index=X.index, columns=['dist'])
    print(f'df_dists_head:\n{df_dists.head()}')
    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:5]
    print(f'df_nn:\n{df_nn}')
    counter = Counter(y[df_nn.index])
    print(f'counter_most_common={counter.most_common()[0][0]}')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_hat_test = knn_predict(X_train, X_test, y_train, 5, 1)
    print(f'{y_hat_test=}')
    acc_score = accuracy_score(y_test, y_hat_test)
    print(f'{acc_score=}')
