import numpy as np

TYPE_CATEGORICAL = 0
TYPE_NUMERICAL = 1

class DecisionTree:
    def __init__(self, max_depth=10, attr_types=None):
        self.root = None
        self.feature_types = np.array(attr_types) if attr_types else None
        self.max_depth = max_depth
        self.train_X, self.train_y = None, None

    # a heuristic to try and determine whether variable is categorical or numerical
    def _infer_types(self):
        self.feature_types = []
        for idx_attr in range(np.size(self.train_X, axis=1)):

            if isinstance(self.train_X[0, idx_attr], str):
                self.feature_types.append(TYPE_CATEGORICAL)
                continue

            if isinstance(self.train_X[0, idx_attr], float):
                self.feature_types.append(TYPE_NUMERICAL)
                continue

            if np.size(set(self.train_X[:, idx_attr])) / np.size(self.train_X) < 0.05:
                self.feature_types.append(TYPE_CATEGORICAL)
                continue

            # default to numerical
            self.feature_types.append(TYPE_NUMERICAL)

        self.feature_types = np.array(self.feature_types)

    def _gini_impurity(self, labels):
        poss_classes = set(labels)
        # 1 - sum_{poss_classes} (p_class ^ 2)
        gimp = 1.0

        for class_ in poss_classes:
            gimp -= (np.sum(labels == class_) / np.size(labels, axis=0)) ** 2

        return gimp

    # feat_type is an element of {TYPE_CATEGORICAL, TYPE_NUMERICAL}
    def _gini_res(self, col_subset, feat_type, target_subset):
        uniq_feat_vals = set(col_subset)

        best_gini, best_thresh = np.inf, np.inf
        res_gini = 0

        if feat_type == TYPE_CATEGORICAL:
            # go over all unique values in a column containing CATEGORICAL values and check the impurity of resulting subsets
            for uniq_val in uniq_feat_vals:
                curr_mask = (col_subset == uniq_val)
                # P(column == uniq_val) * gini(labels[column == uniq_val]) +
                # P(column != uniq_val) * gini(labels[column != uniq_val])
                res_gini = (np.sum(curr_mask) / np.size(col_subset, axis=0)) * self._gini_impurity(target_subset[curr_mask]) + \
                           (np.sum(np.logical_not(curr_mask)) / np.size(col_subset, axis=0)) * self._gini_impurity(target_subset[np.logical_not(curr_mask)])

                if res_gini < best_gini:
                    best_gini = res_gini
                    best_thresh = uniq_val
        else:
            # go over all unique values in a column containing NUMERICAL values and check the impurity of resulting subsets
            # TODO: this can be optimized
            for uniq_val in uniq_feat_vals:
                res_gini = 0

                curr_mask = (col_subset > uniq_val)
                # P(column > uniq_val) * gini(labels[column > uniq_val]) +
                # P(column <= uniq_val) * gini(labels[column <= uniq_val])
                res_gini = (np.sum(curr_mask) / np.size(col_subset, axis=0)) * self._gini_impurity(target_subset[curr_mask]) + \
                           (np.sum(np.logical_not(curr_mask)) * self._gini_impurity(target_subset[np.logical_not(curr_mask)]))

                if res_gini < best_gini:
                    best_gini = res_gini
                    best_thresh = uniq_val

        return (best_thresh, best_gini)

    def fit(self, input_train, labels_train):
        # convert everything to numpy arrays to ease indexing and calculation
        self.train_X = np.array(input_train) if not isinstance(input_train, np.ndarray) else input_train

        self.train_y = np.array(labels_train) if not isinstance(labels_train, np.ndarray) else labels_train

        # using a few heuristics try to find out if attributes are categorical (discrete) or numerical (continuous)
        if not self.feature_types:
            self._infer_types()

        self.root = self._split_rec(self.train_X, self.train_y, 0)

    # recursive function for constructing the tree
    def _split_rec(self, curr_subset_X, curr_subset_y, curr_depth):
        print("[_split_rec()] curr_depth: %d..." % curr_depth)

        # achieved pure subset
        if len(set(curr_subset_y)) == 1:
            print("[_split_rec()] backtracking on depth %d because pure subset has been achieved (%d samples)!" % (curr_depth, np.size(curr_subset_y)))
            return LeafTreeNode({curr_subset_y[0]: 1.0}, curr_subset_y[0], curr_depth)

        # achieved max depth
        if curr_depth == self.max_depth:
            print("[_split_rec()] backtracking on depth %d because max depth has been reached!" % curr_depth)
            uniques, counts = np.unique(curr_subset_y, return_counts=True)
            probs = dict(zip(uniques, counts / np.sum(counts)))
            # value that occurs most frequently
            outcome = uniques[np.argmax(counts)]

            return LeafTreeNode(probs, outcome, curr_depth)



        # 'idx_best_attr' is index of attribute on which the split of current dataset will be performed
        best_gini_gain, idx_best_attr, best_thresh = -np.inf, None, np.inf
        prior_gini = self._gini_impurity(curr_subset_y)

        # find best attribute to split current dataset on
        for idx_attr in range(len(self.feature_types)):
            curr_thresh, curr_res_gini = self._gini_res(curr_subset_X[:, idx_attr], self.feature_types[idx_attr], curr_subset_y)

            curr_gini_gain = prior_gini - curr_res_gini
            if curr_gini_gain > best_gini_gain:
                best_gini_gain = curr_gini_gain
                idx_best_attr = idx_attr
                best_thresh = curr_thresh

        # worse or equal subsets than before
        if best_gini_gain <= 0:
            uniques, counts = np.unique(curr_subset_y, return_counts=True)
            probs = dict(zip(uniques, counts / np.sum(counts)))
            # value that occurs most frequently
            outcome = uniques[np.argmax(counts)]

            return LeafTreeNode(probs, outcome, curr_depth)

        node = InternalTreeNode(idx_best_attr, best_thresh, self.feature_types[idx_best_attr], curr_depth)
        print("[_split_rec()] Best attr index: %d, best gini gain: %.5f, attribute type: %s"
              % (idx_best_attr, best_gini_gain,
                 "CATEGORICAL" if self.feature_types[idx_best_attr] == TYPE_CATEGORICAL else "NUMERICAL"))

        # for categorical values, we use EQ (left branch) and NEQ (right branch) to split, whereas for numerical values,
        # we use GT (left branch) and LTE (right branch) to split
        mask = (curr_subset_X[:, idx_best_attr] == best_thresh) if self.feature_types[idx_best_attr] == TYPE_CATEGORICAL \
            else (curr_subset_X[:, idx_best_attr] > best_thresh)

        # recursively keep splitting the dataset
        node.lchild = self._split_rec(curr_subset_X[mask, :],
                                      curr_subset_y[mask], curr_depth + 1)
        node.rchild = self._split_rec(curr_subset_X[np.logical_not(mask), :],
                                      curr_subset_y[np.logical_not(mask)], curr_depth + 1)

        return node

    def _predict_single(self, input_features, node, verbose=False):
        if isinstance(node, LeafTreeNode):

            if verbose:
                print("[_predict_single()] Reached leaf: outcome %s" % str(node.outcome))

            return node.outcome

        curr_thresh = node.split_val

        if node.split_attr_type == TYPE_CATEGORICAL:

            if input_features[node.split_attr_idx] == curr_thresh:
                if verbose:
                    print("[_predict_single()] [%s] == [%s]" % (str(input_features[node.split_attr_idx]), str(curr_thresh)))

                return self._predict_single(input_features, node.lchild, verbose)
            else:
                if verbose:
                    print("[_predict_single()] [%s] != [%s]" % (str(input_features[node.split_attr_idx]), str(curr_thresh)))

                return self._predict_single(input_features, node.rchild, verbose)

        else:

            if input_features[node.split_attr_idx] > curr_thresh:
                if verbose:
                    print("[_predict_single()] [%s] > [%s]" % (str(input_features[node.split_attr_idx]), str(curr_thresh)))

                return self._predict_single(input_features, node.lchild, verbose)
            else:
                if verbose:
                    print("[_predict_single()] [%s] <= [%s]" % (str(input_features[node.split_attr_idx]), str(curr_thresh)))

                return self._predict_single(input_features, node.rchild, verbose)

    def predict(self, data, verbose=False):
        return np.array([self._predict_single(row, self.root, verbose) for row in data])

    # breadth-first traversal of decision tree + printing
    def _traverse(self, node):
        queue = [node]

        while len(queue) != 0:
            curr_node = queue.pop(0)

            if isinstance(curr_node, InternalTreeNode):
                if curr_node.lchild:
                    queue.append(curr_node.lchild)
                if curr_node.rchild:
                    queue.append(curr_node.rchild)
            else:
                print(curr_node)

    # caller method for breadth-first traversal of tree + printing
    def traverse(self):
        self._traverse(self.root)

class InternalTreeNode:
    def __init__(self, attr_idx, split_val, attr_type, depth):
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
        # {class: probability}
        self.probabilities = probs
        self.outcome = outcome
        self.depth = depth

    def __str__(self):
        return "[TERMINAL at depth %d] %s" % (self.depth, str(self.probabilities))