class RegressionTree:
  class Node:
    def __init__(self,feature_i = None, right_child=None, left_child = None, threshold= None, result = None):
      self.feature_i = feature_i
      self.right_child = right_child
      self.left_child = left_child
      self.threshold = threshold
      self.result = result


  def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
      self.min_samples_split = min_samples_split
      self.min_impurity = min_impurity
      self.max_depth = max_depth
      self.root = None
  def TSE(self,y):
    return ((y-np.mean(y))**2).sum()
  def b_tree(self, x, y, depth):

    if (depth >= self.max_depth or len(y) < self.min_samples_split or self.TSE(y) < self.min_impurity):
      return self.Node(result=np.mean(y))

    parent_error = self.TSE(y)

    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature_i in range(x.shape[1]):
        thresholds = np.unique(x[:, feature_i])

        for threshold in thresholds:
            l_mask = x[:, feature_i] < threshold
            r_mask = ~l_mask

            l_y = y[l_mask]
            r_y = y[r_mask]

            if len(l_y) == 0 or len(r_y) == 0:
                continue

            child_error = self.TSE(l_y) + self.TSE(r_y)
            gain = parent_error - child_error

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_i
                best_threshold = threshold

    if best_gain <= 0:
        return self.Node(result=np.mean(y))

    l_mask = x[:, best_feature] < best_threshold
    r_mask = ~l_mask

    left_child = self.b_tree(x[l_mask], y[l_mask], depth + 1)
    right_child = self.b_tree(x[r_mask], y[r_mask], depth + 1)

    return self.Node(
        feature_i=best_feature,
        threshold=best_threshold,
        left_child=left_child,
        right_child=right_child
    )

  def fit(self, X, y):
    y = np.array(y)
    X = np.array(X)
    self.root = self.b_tree(X, y, 0)

  def predict(self, X):
    x = np.array(X)
    arr_result = []
    for row in x:
      arr_result.append(self.x_check(row,self.root))
    return np.array(arr_result)
