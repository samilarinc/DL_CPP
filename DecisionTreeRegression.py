import numpy as np


class Decision_Tree_Regression(object):
    def __init__(self, max_depth=5, sample_split_thres=2):
        self.sample_split_thres = sample_split_thres
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self.__construct(X, y)
    
    def predict(self, X):
        return np.array([self.__find_x_from_node(x, self.root) for x in X])

    def __construct(self, X, y, curr_depth = 0):
        sample_num = np.shape(X)[0]
        if sample_num >= self.sample_split_thres and curr_depth <= self.max_depth:
            best_feature, best_thres = self.__find_best_split(X, y)
            left_samples, right_samples = self.__split_left_right(X[:, best_feature], best_thres)
            left = self.__construct(X[left_samples, :], y[left_samples], curr_depth + 1)
            right = self.__construct(X[right_samples, :], y[right_samples], curr_depth + 1)
            return Node(best_feature, best_thres, left, right) # internal node -> value is None
        leaf_val = np.mean(y)  # mean of the target values
        return Node(value = leaf_val) # leaf node -> other attributes are None

    def __find_best_split(self, X, y):
        feature_num = np.shape(X)[1]
        best_feature, best_thres = None, None
        min_impurity = float('inf') # we want to minimize the impurity
        for feature_ind in range(feature_num):
            thresholds = np.unique(X[:, feature_ind])
            for thres in thresholds:
                impurity = self.__calculate_impurity(y, thres, X[:, feature_ind])
                if impurity < min_impurity:
                    min_impurity = impurity
                    best_feature = feature_ind
                    best_thres = thres
        return best_feature, best_thres # these indicate the best way to split the data

    def __calculate_impurity(self, y, thres, feature):
        left_samples, right_samples = self.__split_left_right(feature, thres)
        if len(left_samples) == 0 or len(right_samples) == 0:
            return float('inf')
        left_labels = y[left_samples]
        right_labels = y[right_samples]
        p_left = len(left_labels) / len(y)  # proportion of the left subtree
        p_right = 1 - p_left  # proportion of the right subtree
        impurity = p_left * self.__calculate_gini(left_labels) + p_right * self.__calculate_gini(right_labels)
        return impurity

    def __calculate_gini(self, y):
        uniques, counts = np.unique(y, return_counts=True)
        p_classes = counts / len(y) # proportion of each class
        return 1 - np.sum(np.square(p_classes))  # gini impurity
        # gini := 1 - sigma_sum (p_i^2)

    def __split_left_right(self, feature, thres):
        # indices of the left subtree
        left_samples = np.argwhere(feature <= thres).flatten()
        # indices of the right subtree
        right_samples = np.argwhere(feature > thres).flatten()
        return left_samples, right_samples


    def __find_x_from_node(self, x, node):
        if node.value is not None:  # leaf node
            return node.value  # return the value of the leaf node
        feature_value = x[node.feature_ind]  # get the value of the feature
        if feature_value <= node.thres:  # if <= thres, go left
            return self.__find_x_from_node(x, node.left)
        return self.__find_x_from_node(x, node.right)


class Node(object):
    def __init__(self, feature_ind=None, thres=None, left=None, right=None, value=None):
        self.feature_ind = feature_ind
        self.thres = thres
        self.left = left
        self.right = right
        self.value = value

    def __repr__(self):
        if self.value is None:
            return "Internal(feature_ind=%s, thres=%f)"%(self.feature_ind, self.thres)
        return "Leaf(value=%f)"%(self.value)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

## Example Usage

# model = Decision_Tree_Regression(max_depth, sample_split_threshold)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_val)
# mse = mean_squared_error(y_val, y_pred)