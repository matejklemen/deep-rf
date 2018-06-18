import numpy as np

TYPE_CATEGORICAL = 0
TYPE_NUMERICAL = 1

# finds where elements of array 'first' are in array 'second'
# warning: very expensive in terms of memory!
# found at: https://stackoverflow.com/a/40449296
def find_reordering(first, second):
    return np.where(second[:, None] == first[None, :])[0]

class DecisionTree:
    def __init__(self, max_depth=10,
                 attr_types=None,
                 classes_=None,
                 random_state=None,
                 extremely_randomized=False,
                 max_features=None):
        """
        :param max_depth:
        :param attr_types: a list/np.ndarray representing attribute types that will be passed into fit() (and predict()).
            Most likely not needed, but the heuristic for determining attribute types may sometimes fail (if very raw
            data is passed in).
        :param classes_: a list/np.ndarray representing which index will represent which class in the probability vector
        :param random_state: an integer determining the random state for random number generator.
        :param extremely_randomized: a boolean determining whether to build an extremely randomized decision tree
        :param max_features: number of features that are taken into account when choosing the best attribute to split
            the data set on. Default value max_features=None, which means that all the features will be considered.
            If it is not None, 'max_features' features will be randomly sampled (without replacement).
        """
        self.root = None
        self.max_depth = max_depth

        self.classes_ = np.array(classes_) if classes_ is not None else None
        self.feature_types = np.array(attr_types) if attr_types else None

        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

        self.extremely_randomized = extremely_randomized
        self.max_features = max_features

    # a heuristic to try and determine whether variable is categorical or numerical
    def _infer_types(self, train_X):
        self.feature_types = []
        for idx_attr in range(train_X.shape[1]):

            if isinstance(train_X[0, idx_attr], str):
                self.feature_types.append(TYPE_CATEGORICAL)
                continue

            if isinstance(train_X[0, idx_attr], float):
                self.feature_types.append(TYPE_NUMERICAL)
                continue

            # less than 5% of all values in current column are unique values (= most likely a categorical attr.)
            if np.unique(train_X[:, idx_attr]).shape[0] / train_X.shape[0] < 0.05:
                self.feature_types.append(TYPE_CATEGORICAL)
                continue

            # default to numerical
            self.feature_types.append(TYPE_NUMERICAL)

        self.feature_types = np.array(self.feature_types)

    def _assign_labels(self, labels_train):
        self.classes_ = np.unique(labels_train)

    def _gini_impurity(self, labels):
        """ Calculates gini impurity: 1 - sum_{poss_classes} (p_class ^ 2) """
        return 1 - np.sum(np.square(np.unique(labels, return_counts=True)[1] / labels.shape[0]))

    # feat_type is an element of {TYPE_CATEGORICAL, TYPE_NUMERICAL}
    def _gini_res(self, col_subset, feat_type, target_subset, curr_thresh):

        size_colsub = col_subset.shape[0]

        # if feature is categorical, use '>' and '<=' for splitting, else use '==' and '!='
        curr_mask = (col_subset == curr_thresh) if feat_type == TYPE_CATEGORICAL else (col_subset > curr_thresh)

        # probability that the condition is true
        ptrue = np.sum(curr_mask) / size_colsub

        res_gini = ptrue * self._gini_impurity(target_subset[curr_mask]) + \
                   (1 - ptrue) * self._gini_impurity(target_subset[np.logical_not(curr_mask)])

        return res_gini

    def fit(self, input_train, labels_train):
        """ Fit decision tree according to 'input_train' and 'labels_train'
        :param input_train:
        :param labels_train:
        :return: None (builds the tree in-place).
        """
        # convert everything to numpy arrays to ease indexing and calculation
        if not isinstance(input_train, np.ndarray):
            input_train = np.array(input_train)

        if not isinstance(labels_train, np.ndarray):
            labels_train = np.array(labels_train)

        # assign mapping from class label to index in probability vector
        if self.classes_ is None:
            self._assign_labels(labels_train)

        # using a few heuristics try to find out if attributes are categorical (discrete) or numerical (continuous)
        if self.feature_types is None:
            self._infer_types(input_train)

        # if number of features is not specified, consider all features
        if self.max_features is None:
            self.max_features = self.feature_types.shape[0]

        self.root = self._split_rec(input_train, labels_train, 0, self.max_features)

    # recursive function for constructing the tree
    def _split_rec(self, curr_subset_X, curr_subset_y, curr_depth, num_features):

        # achieved pure subset
        if np.unique(curr_subset_y).shape[0] == 1:
            probs = np.zeros((1, self.classes_.shape[0]))

            # put probability at right place in our internal representation
            idx_class = np.where(self.classes_ == curr_subset_y[0])[0][0]
            probs[0, idx_class] = 1.0

            return LeafTreeNode(probs, curr_subset_y[0], curr_depth)

        # achieved max depth
        if curr_depth == self.max_depth:
            uniques, counts = np.unique(curr_subset_y, return_counts=True)
            probs = np.zeros((1, self.classes_.shape[0]))
            subset_size = curr_subset_y.shape[0]

            # need to find correct places to put probability in our internal representation
            right_places = find_reordering(uniques, self.classes_)

            probs[0, right_places] = counts / subset_size

            # value that occurs most frequently
            outcome = uniques[np.argmax(counts)]

            return LeafTreeNode(probs, outcome, curr_depth)

        # best values for all (chosen) attributes
        best_gini_gain, idx_best_attr, best_thresh = -np.inf, None, np.inf
        prior_gini = self._gini_impurity(curr_subset_y)

        # if we do not want to take into account all of the features (i.e. 'num_features' < number of all attributes),
        # then randomly choose 'num_features' of them without replacement
        chosen_features = range(num_features) if num_features == self.feature_types.shape[0] \
            else np.random.choice(self.feature_types.shape[0], num_features, replace=False)

        # find best attribute to split current data set on
        for idx_attr in chosen_features:
            all_thresholds = np.unique(curr_subset_X[:, idx_attr])

            """
                In extremely randomized decision trees, a random threshold value is selected for each chosen feature,
                among which the best (according to score function, e.g. gini) threshold is selected and used for
                splitting the data set.
            """
            selected_thresholds = np.random.choice(all_thresholds, 1) \
                if self.extremely_randomized else all_thresholds

            # best values for current attribute
            curr_best_res_gini, curr_best_thresh = prior_gini, None

            # find best split for current attribute
            for thresh in selected_thresholds:
                res_gini = self._gini_res(curr_subset_X[:, idx_attr], self.feature_types[idx_attr], curr_subset_y, thresh)

                if res_gini < curr_best_res_gini:
                    curr_best_res_gini = res_gini
                    curr_best_thresh = thresh

                # if residual gini is 0, we found best possible split (no point in searching for other splits)
                if res_gini == 0:
                    break

            curr_gini_gain = prior_gini - curr_best_res_gini
            if curr_gini_gain > best_gini_gain:
                best_gini_gain = curr_gini_gain
                idx_best_attr = idx_attr
                best_thresh = curr_best_thresh

            # found attribute + threshold combination which makes a pure subset - there might be other combinations
            # as well, but the first one is good enough (and possibly speeds things up)
            if curr_gini_gain == prior_gini:
                break

        # worse or equal subsets than before
        if best_gini_gain <= 0:
            uniques, counts = np.unique(curr_subset_y, return_counts=True)
            probs = np.zeros((1, self.classes_.shape[0]))

            subset_size = curr_subset_y.shape[0]

            # need to find correct places to put probability in our internal representation
            right_places = find_reordering(uniques, self.classes_)

            probs[0, right_places] = counts / subset_size

            # value that occurs most frequently
            outcome = uniques[np.argmax(counts)]

            return LeafTreeNode(probs, outcome, curr_depth)

        node = InternalTreeNode(idx_best_attr, best_thresh, self.feature_types[idx_best_attr], curr_depth)

        # for categorical values, we use EQ (left branch) and NEQ (right branch) to split, whereas for numerical values,
        # we use GT (left branch) and LTE (right branch) to split
        mask = (curr_subset_X[:, idx_best_attr] == best_thresh) if self.feature_types[idx_best_attr] == TYPE_CATEGORICAL \
            else (curr_subset_X[:, idx_best_attr] > best_thresh)

        # recursively keep splitting the data set
        node.lchild = self._split_rec(curr_subset_X[mask, :], curr_subset_y[mask],
                                      curr_depth + 1, num_features)
        node.rchild = self._split_rec(curr_subset_X[np.logical_not(mask), :], curr_subset_y[np.logical_not(mask)],
                                      curr_depth + 1, num_features)

        return node

    def _predict_single(self, input_features, node):
        """
        Predicts a single instance - returns probabilities!
        :param input_features:
        :param node:
        :return: class label for a single instance or (if return_probabilities=True) probability vector for single
            instance.
        """
        if isinstance(node, LeafTreeNode):
            return node.probabilities[0]

        curr_thresh = node.split_val

        if node.split_attr_type == TYPE_CATEGORICAL:
            if input_features[node.split_attr_idx] == curr_thresh:
                return self._predict_single(input_features, node.lchild)
            else:
                return self._predict_single(input_features, node.rchild)

        else:
            if input_features[node.split_attr_idx] > curr_thresh:
                return self._predict_single(input_features, node.lchild)
            else:
                return self._predict_single(input_features, node.rchild)

    def predict(self, data):
        """
        Predict output for new data - returns labels.
        :param data:
        :return: class labels
        """
        class_idx = np.argmax(self.predict_proba(data), axis=1)
        return np.array([self.classes_[idx] for idx in class_idx])

    def predict_proba(self, data):
        """
        Predict output for new data - returns probabilities for classes, whose indices are determined by
        self.idx_label_mapping.
        :param data:
        :return: class probabilities
        """
        return np.array([self._predict_single(row, self.root) for row in data])


class InternalTreeNode:
    def __init__(self, attr_idx, split_val, attr_type, depth):
        """
        :param attr_idx: index of the attribute that is used for splitting the data set in current node.
        :param split_val: threshold value for splitting the data set.
        :param attr_type: defines whether the attribute used in current node is categorical (TYPE_CATEGORICAL) or
            numerical (TYPE_NUMERICAL).
        :param depth: depth that the current node is on (might actually be redundant).
        """
        self.split_attr_idx = attr_idx
        self.split_val = split_val
        self.split_attr_type = attr_type
        self.depth = depth
        self.lchild, self.rchild = None, None

    def __str__(self):
        return "[depth %d, attr %d, thresh: %3f]" % (self.depth, self.split_attr_idx, self.split_val)


# contains class prediction
class LeafTreeNode:
    def __init__(self, probs, outcome, depth):
        self.probabilities = probs
        self.outcome = outcome
        self.depth = depth

    def __str__(self):
        return "[TERMINAL at depth %d] %s" % (self.depth, str(self.probabilities))