class SVR:
    def __init__(self, epsilon=0.1, C=1, kernel_name='linear', power=2, gamma=None, coef=2):
        self.epsilon = epsilon
        self.C = C
        self.kernel_name = kernel_name
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.w = None
        self.b = 0

    def fit(self, X, y, lr=0.01, epochs=1000):
        X = np.array(X)
        y = np.array(y)

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(epochs):
            for i in range(n_samples):
                y_pred = np.dot(X[i], self.w) + self.b
                error = y[i] - y_pred

                if error > self.epsilon:
                    self.w += lr * (self.C * X[i] - self.w)
                    self.b += lr * self.C
                elif error < -self.epsilon:
                    self.w += lr * (-self.C * X[i] - self.w)
                    self.b -= lr * self.C
                else:
                    self.w -= lr * self.w

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.w) + self.b





