import numpy as np
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class TreeNode:
    predict_val: float  # prediction of node
    feature: int  # feature idx to split
    threshold: float  # threshold to split

    depth: float  # current node depth

    left = None
    right = None


class Criterion:
    def __init__(self, mu, k):
        self.mu = mu
        self.k = k

    def __call__(self, y, prev_pred_sum=None):
        pred = self.pred(y, prev_pred_sum)
        mse = np.mean((y - pred) ** 2)

        if self.k == 0:
            return mse
        
        var = np.mean(pred ** 2 - 2 * pred * prev_pred_sum / self.k)

        return mse * (1 - self.mu) / (self.k + 1) - var * self.mu * self.k / (self.k + 1) ** 2
    
    def pred(self, y, prev_pred_sum=None):
        if self.k == 0:
            return y.mean()

        val = self.k + 1 - self.mu * (2 * self.k + 1)
        if abs(val) < 1e-5:
            print(val, self.k, self.mu)
        return ((self.k + 1) * (1 - self.mu) * np.mean(y) - self.k * self.mu * np.mean(prev_pred_sum / self.k)) / (self.k + 1 - self.mu * (2 * self.k + 1))

        return (((1 - self.mu) / (self.k + 1) * np.mean(y)
                - 2 * self.mu * self.k / ((self.k + 1) ** 2) * np.mean(prev_pred_sum / self.k))
                / ((1 - self.mu) / (self.k + 1)
                   - 2 * self.mu * self.k / ((self.k + 1) ** 2)))

    def get_threshold(self, feature, y, prev_pred_sum=None, min_samples_leaf=1):
        threshold_ind = np.argsort(feature)
        feature_srt = feature[threshold_ind]
        y_srt = y[threshold_ind]
        prev_pred_sum_srt = prev_pred_sum[threshold_ind]

        _, threshold_ind_unique = np.unique(feature_srt, return_index=True)

        loss_best = None
        threshold_best = None

        for threshold in threshold_ind_unique:
            if threshold >= min_samples_leaf and len(y) - threshold >= min_samples_leaf:
                loss_left = self.__call__(y_srt[:threshold], prev_pred_sum_srt[:threshold])
                loss_right = self.__call__(y_srt[threshold:], prev_pred_sum_srt[threshold:])

                frac_left = threshold / len(y)
                frac_right = 1 - frac_left
                loss = frac_left * loss_left + frac_right * loss_right

                if (loss_best is None) or (loss < loss_best):
                    loss_best = loss
                    threshold_best = feature_srt[threshold - 1]

        return threshold_best, loss_best


class DecisionTree:
    def __init__(self, criterion, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def _fit(self, X: np.ndarray, y: np.ndarray, prev_pred_sum: np.ndarray, indices: np.ndarray, depth: int):
        if (self.max_depth is None or depth < self.max_depth) and self.min_samples_split < indices.size:
            loss_best = None
            feature_best = None
            threshold_best = None

            if self.max_features is None:
                features = range(X.shape[1])
            else:
                features = np.random.choice(X.shape[1], self.max_features, replace=False)

            for feature in features:
                threshold, loss = self.criterion.get_threshold(X[indices, feature], y[indices],
                                                               prev_pred_sum[indices],
                                                               min_samples_leaf=self.min_samples_leaf)

                if loss_best is None or (loss is not None and loss < loss_best):
                    loss_best = loss
                    feature_best = feature
                    threshold_best = threshold

            if loss_best is None:
                return TreeNode(predict_val=self.criterion.pred(y[indices], prev_pred_sum[indices]), feature=None, threshold=None, depth=depth)

            node = TreeNode(predict_val=self.criterion.pred(y[indices], prev_pred_sum[indices]), feature=feature_best, threshold=threshold_best,
                            depth=depth)

            node.left = self._fit(X, y, prev_pred_sum, indices[X[indices, feature_best] <= threshold_best], depth + 1)
            node.right = self._fit(X, y, prev_pred_sum, indices[X[indices, feature_best] > threshold_best], depth + 1)
        else:
            node = TreeNode(predict_val=self.criterion.pred(y[indices], prev_pred_sum[indices]), feature=None, threshold=None, depth=depth)

        return node

    def fit(self, X, y, prev_pred_sum=None):
        indices = np.arange(X.shape[0])
        self.root = self._fit(X, y, prev_pred_sum, indices, 0)

    def _predict(self, X: np.ndarray, predictions: np.ndarray, indices: np.ndarray, node: TreeNode):
        if node.feature is not None:
            self._predict(X, predictions, indices[X[indices, node.feature] <= node.threshold], node.left)
            self._predict(X, predictions, indices[X[indices, node.feature] > node.threshold], node.right)
        else:
            predictions[indices] = node.predict_val

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        indices = np.arange(X.shape[0])
        self._predict(X, predictions, indices, self.root)
        return predictions


class RandomForest:
    def __init__(self, n_estimators, mu, **kwargs):
        self.n_estimators = n_estimators
        self.mu = mu
        self.kwargs = kwargs
        self.trees = []

    def fit(self, X, y):
        prev_pred_sum = np.zeros_like(y)
        self.trees = []
        for i in range(self.n_estimators):
            indices = np.random.choice(X.shape[0], X.shape[0])
            criterion = Criterion(self.mu, i)
            tree = DecisionTree(criterion, **self.kwargs)
            tree.fit(X[indices], y[indices], prev_pred_sum)
            prev_pred_sum += tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        preds = [tree.predict(X) for tree in self.trees]
        return np.mean(preds, axis=0)
