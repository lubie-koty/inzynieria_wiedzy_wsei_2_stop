import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def tstr(n: int) -> str:
    return n * '\t'


if __name__ == '__main__':
    iris_dataset = load_iris()
    X = iris_dataset.data
    y = iris_dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    classifier = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
    classifier.fit(X_train, y_train)

    n_nodes = classifier.tree_.node_count
    children_left = classifier.tree_.children_left
    children_right = classifier.tree_.children_right
    feature = classifier.tree_.feature
    threshold = classifier.tree_.threshold

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]

    while len(stack) > 0:
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    print(
        f'The binary tree structure has {n_nodes} nodes and has' + \
        'the following tree structure:'
    )

    for i in range(n_nodes):
        if is_leaves[i]:
            print(f'{tstr(node_depth[i])}node={i} is a leaf node.')
        else:
            print(
                f'{tstr(node_depth[i])}node={i} is a split node: ' + \
                f'go to node {children_left[i]} if X[:, {feature[i]}] <= {threshold[i]} ' + \
                f'else to node {children_right[i]}.'
            )
    
    tree.plot_tree(classifier)
    plt.show()

    node_indicator = classifier.decision_path(X_test)
    leaf_id = classifier.apply(X_test)

    sample_id = 0
    node_index = node_indicator.indices[
        node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
    ]

    print(f'Rules used to predict sample {sample_id}:')
    for node_id in node_index:
        if leaf_id[sample_id] == node_id:
            continue

        threshold_sign = '<=' if X_test[sample_id, feature[node_id]] <= threshold[node_id] else '>'

        print(
            f'decision node {node_id} : (X_test[{sample_id}, {feature[node_id]}]' + \
            f'= {X_test[sample_id, feature[node_id]]}) {threshold_sign} {threshold[node_id]}'
        )        

    sample_ids = [0, 1]

    common_nodes = node_indicator.toarray()[sample_ids].sum(axis=0) == len(sample_ids)
    common_node_id = np.arange(n_nodes)[common_nodes]

    print(f'The following samples {sample_ids} share the node(s) {common_node_id} in the tree.')
    print(f'This is {100 * len(common_node_id) / n_nodes}% of all nodes.')
