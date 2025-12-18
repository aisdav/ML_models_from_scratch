class GDLogisticRegression:
    def __init__(self, learning_rate=0.1, tolerance=1e-4, max_iter=1000):
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iter = max_iter

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.ravel()
        self.loss_history=[]

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.max_iter):
            z = X @ self.weights + self.bias
            y_hat = self.sigmoid(z)
            loss = -np.mean(
            y * np.log(y_hat + 1e-15) +
                (1 - y) * np.log(1 - y_hat + 1e-15)
            )
            self.loss_history.append(loss)
            dw = (1 / n_samples) * X.T @ (y_hat - y)
            db = (1 / n_samples) * np.sum(y_hat - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if np.linalg.norm(dw) < self.tolerance:
                break

    def predict_proba(self, X):
        z = X @ self.weights + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
