import numpy as np
import pandas as pd
from dataclasses import dataclass
from copy import deepcopy


class DecisionTreeCART:
    def __init__(
        self,
        max_depth=10,
        min_samples=2,
        ccp_alpha=0.0,
        regression=False
    ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.ccp_alpha = ccp_alpha
        self.regression = regression
        self.root = None


    @staticmethod
    def _gini(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    @staticmethod
    def _mse(y):
        return np.mean((y - y.mean()) ** 2)


    def _split_loss(self, y_left, y_right, criterion):
        n = len(y_left) + len(y_right)
        return (
            len(y_left) / n * criterion(y_left)
            + len(y_right) / n * criterion(y_right)
        )

    def _best_split(self, X, y, criterion):
        best_feature, best_threshold = None, None
        best_loss = np.inf

        for feature in X.columns:
            values = np.sort(X[feature].unique())

            for i in range(1, len(values)):
                threshold = (values[i] + values[i - 1]) / 2
                left_mask = X[feature] <= threshold
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                loss = self._split_loss(
                    y[left_mask],
                    y[right_mask],
                    criterion
                )

                if loss < best_loss:
                    best_loss = loss
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold


    def _build_tree(self, X, y, depth):
        criterion = self._mse if self.regression else self._gini
        n_samples = len(y)
        Rt = criterion(y)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples
            or np.unique(y).size == 1
        ):
            prediction = y.mean() if self.regression else y.mode().iloc[0]
            return Node(
                is_leaf=True,
                prediction=prediction,
                Rt=Rt,
                n_samples=n_samples
            )

        feature, threshold = self._best_split(X, y, criterion)

        if feature is None:
            prediction = y.mean() if self.regression else y.mode().iloc[0]
            return Node(True, prediction, Rt=Rt, n_samples=n_samples)

        left_mask = X[feature] <= threshold
        right_mask = ~left_mask

        left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        prediction = y.mean() if self.regression else y.mode().iloc[0]

        return Node(
            is_leaf=False,
            prediction=prediction,
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            Rt=Rt,
            n_samples=n_samples
        )



    def _subtree_error(self, node):
        if node.is_leaf:
            return node.Rt, 1

        left_err, left_leaves = self._subtree_error(node.left)
        right_err, right_leaves = self._subtree_error(node.right)

        return left_err + right_err, left_leaves + right_leaves

    def _ccp_alpha(self, node):
        if node.is_leaf:
            return np.inf, None

        subtree_Rt, leaves = self._subtree_error(node)
        alpha = (node.Rt - subtree_Rt) / (leaves - 1)

        left_alpha, left_node = self._ccp_alpha(node.left)
        right_alpha, right_node = self._ccp_alpha(node.right)

        min_alpha, min_node = alpha, node

        if left_alpha < min_alpha:
            min_alpha, min_node = left_alpha, left_node
        if right_alpha < min_alpha:
            min_alpha, min_node = right_alpha, right_node

        return min_alpha, min_node

    def _prune(self, root):
        while True:
            alpha, node = self._ccp_alpha(root)
            if alpha > self.ccp_alpha or node is None:
                break
            node.is_leaf = True
            node.left = None
            node.right = None
        return root

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.root = self._build_tree(X, y, depth=0)
        if self.ccp_alpha > 0:
            self.root = self._prune(self.root)

    def _predict_one(self, row, node):
        if node.is_leaf:
            return node.prediction
        if row[node.feature] <= node.threshold:
            return self._predict_one(row, node.left)
        return self._predict_one(row, node.right)

    def predict(self, X: pd.DataFrame):
        return np.array([self._predict_one(row, self.root) for _, row in X.iterrows()])
