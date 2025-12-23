class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes)

        self.priors = np.zeros(n_classes)
        self.means = np.zeros((n_classes, n_features))
        self.stds = np.zeros((n_classes, n_features))

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.priors[idx] = X_c.shape[0] / n_samples
            self.means[idx] = X_c.mean(axis=0)
            self.stds[idx] = X_c.std(axis=0) + 1e-9  # защита от деления на 0

    def _gaussian_pdf(self, x, mean, std):
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

    def predict(self, X):
        predictions = []

        for x in X:
            posteriors = []

            for idx, c in enumerate(self.classes):
                prior = np.log(self.priors[idx])

                likelihood = np.sum(
                    np.log(self._gaussian_pdf(x, self.means[idx], self.stds[idx]))
                )

                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes[np.argmax(posteriors)])

        return np.array(predictions)
