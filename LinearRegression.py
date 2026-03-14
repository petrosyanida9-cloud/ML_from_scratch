import numpy as np
import progressbar

widgets = ['Model Training: ', progressbar.Percentage(), ' ',
            progressbar.Bar(marker="-", left="[", right="]"),
            ' ', progressbar.ETA()]

class MyLinearRegression:

  def __init__(self, regularization=None, lam=0, learning_rate=1e-3, tol=0.05):

    self.regularization = regularization
    self.lam = lam
    self.learning_rate = learning_rate
    self.tol = tol
    self.weights = None

  def fit(self, X, y):
    X = np.array(X)
    X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
    print("abcc")

    if self.regularization is None:
      self.weights = np.linalg.inv(X.T@X)@X.T@y

    elif self.regularization == 'l2':
      self.weights = np.linalg.inv(X.T@X+self.lam * np.eye(X.shape[1]))@ X.T@y
    elif self.regularization == 'l1':

      self.weights = np.random.randn(X.shape[1])
      print("l1")

      converged = False
      self.loss = []
      i = 0
      while (not converged):
        i += 1
        y_pred = X@self.weights

        self.loss.append(np.mean((y_pred - y) ** 2))
        grad = (1 / len(y)) * X.T @ (y_pred - y)
        grad += self.lam * np.sign(self.weights)
        grad[0] = 0

        new_weights = self.weights - self.learning_rate * grad
        print("converged")
        converged = np.linalg.norm(new_weights-self.weights)< self.tol
        self.weights = new_weights
      print(f'Converged in {i} steps')

  def predict(self, X):
    X = np.array(X)
    print("predict")
    X = np.insert(X, 0, np.ones(X.shape[0]), axis = 1)
    return X @ self.weights
