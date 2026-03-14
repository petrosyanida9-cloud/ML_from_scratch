class LogisticRegression:
    def __init__(self, learning_rate=1e-3, nr_iterations=10, batch_size=64):
        self.learning_rate = learning_rate
        self.nr_iterations = nr_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for j in range(self.nr_iterations):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                linear_output = np.dot(X_batch, self.weights) + self.bias
                y_pred = self._sigmoid(linear_output)

                dw = np.dot(X_batch.T, (y_pred - y_batch)) / len(y_batch)
                db = np.mean(y_pred - y_batch)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.array(X)
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_output)
        return (y_pred >= 0.5).astype(int)

