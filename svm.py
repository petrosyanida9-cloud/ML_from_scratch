import numpy as np
import cvxopt

cvxopt.solvers.options['show_progress'] = False

class SVM:
    def __init__(self, C=1):
        self.C = C
        self.alphas = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.w = None
        self.t = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        K = X @ X.T
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)

        if not self.C:
            G = cvxopt.matrix(-np.eye(n))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.C)))

        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alphas = np.ravel(sol['x'])
        idx = alphas > 1e-7

        self.alphas = alphas[idx]
        self.support_vectors = X[idx]
        self.support_vector_labels = y[idx]

        self.w = ((self.alphas * self.support_vector_labels)[:, None] *
                  self.support_vectors).sum(axis=0)

        self.t = float(self.w @ self.support_vectors[0] -
                       self.support_vector_labels[0])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.sign(self.w @ X.T - self.t)


    def predict(self, X):
       return np.sign(self.w @ X.T - self.t)