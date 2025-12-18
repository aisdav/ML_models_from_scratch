import numpy as np
class GDLinearRegression:
    def __init__(self, learning_rate=0.01, tolerance=1e-8,max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter= max_iter

    def fit(self, X, y):
        y= y.ravel()
        n_samples, n_features = X.shape

        self.bias = 0.0
        self.weights = np.zeros(n_features)
        self.loss_history = []

        previous_db = 0.0
        previous_dw = np.zeros(n_features)

        for _ in range(self.max_iter):
            y_pred = X @ self.weights + self.bias
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            db = (1 / n_samples) * np.sum(y_pred - y)
            dw = (1 / n_samples) * (X.T @ (y_pred - y))

            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw

            if np.abs(db - previous_db) < self.tolerance and \
               np.linalg.norm(dw - previous_dw) < self.tolerance:
                break

            previous_db = db
            previous_dw = dw

    def predict(self, X_test):
        return X_test @ self.weights + self.bias
class MatrixLinearRegression:
    def fit(self,X,y):
        X= np.insert(X,0,1,axis=1)
        XT_X_inv = np.linalg.inv(X.T@X)
        weights =np.linalg.multi_dot([XT_X_inv,X.T,y])
        self.bias,self.weights = weights[0],weights[1:]
    def predict (self,X_test):
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        return X_test @ self.weights + self.bias
