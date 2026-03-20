import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

class GBMClassifier:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3, random_state=0):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

    def _softmax(self, scores):
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.K = len(self.classes_)
        self.trees = {k: [] for k in range(self.K)}

        y_onehot = pd.get_dummies(y).reindex(columns=self.classes_, fill_value=0).to_numpy()
        scores = np.zeros((len(X), self.K))

        for _ in range(self.n_estimators):
            probabilities = self._softmax(scores)

            for k in range(self.K):
                residuals = y_onehot[:, k] - probabilities[:, k]

                tree = DecisionTreeRegressor(
                    criterion='friedman_mse',
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
                tree.fit(X, residuals)
                self.trees[k].append(tree)

                scores[:, k] += self.learning_rate * tree.predict(X)

        return self

    def predict_proba(self, X):
        scores = np.zeros((len(X), self.K))

        for i in range(self.n_estimators):
            for k in range(self.K):
                scores[:, k] += self.learning_rate * self.trees[k][i].predict(X)

        return self._softmax(scores)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes_[class_indices]
