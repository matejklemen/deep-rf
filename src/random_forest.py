import decision_tree
import numpy as np
import time


class RandomForest:
    def __init__(self, n_estimators=100,
                 max_depth=10,
                 attr_types=None,
                 classes_=None,
                 random_state=None,
                 max_features=None,
                 extremely_randomized=False,
                 labels_encoded=False):
        """
        Parameters
        ----------
        :param n_estimators: int
                Number of trees in forest.
        :param max_depth: int
                Max depth for trees, grown in the forest.
        :param attr_types: list or numpy.ndarray (default: None)
                Attribute types that will be passed into fit() (and predict()).
                Likely not needed, but the heuristic for determining attribute types may sometimes fail (if very raw
                data is passed in).
        :param classes_: list or numpy.ndarray (default: None)
                Mapping between classes and indices in the probability vectors.
        :param random_state: int (default: None)
                The random state for random number generator.
        :param max_features: int
                Number of randomly selected features.
        :param extremely_randomized: bool
                Flag, specifying whether to build a completely random forest.
        :param labels_encoded: bool
                Will labels in training set already be encoded as stated in 'classes_'?
        """
        self.num_trees = n_estimators
        self.max_depth = max_depth
        self.trees = []

        self.classes_ = np.array(classes_) if classes_ is not None else None

        self.attr_types = attr_types
        if random_state is not None:
            np.random.seed(random_state)

        self.max_features = max_features
        self.extremely_randomized = extremely_randomized
        self.labels_encoded = labels_encoded

    def _assign_labels(self, labels_train):
        if self.classes_ is None:
            # wanted classes not provided
            self.classes_, encoded_labels = np.unique(labels_train, return_inverse=True)
        else:
            encoded_labels = np.zeros_like(labels_train, np.int32)
            for encoded_label in range(self.classes_.shape[0]):
                encoded_labels[labels_train == self.classes_[encoded_label]] = encoded_label

        return encoded_labels

    def fit(self, input_train, labels_train):
        # assign mapping from class label to index in probability vector
        if not self.labels_encoded:
            labels_train = self._assign_labels(labels_train)

        self.trees = []
        sample_size = input_train.shape[0]
        row_indices = np.arange(sample_size)

        # if number of used features is not specified, default to sqrt(number of all features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(input_train.shape[1]))

        for idx_tree in range(self.num_trees):
            # choose sample for current tree (random with replacement)
            curr_sample_indices = np.random.choice(row_indices, sample_size)

            curr_input = input_train[curr_sample_indices, :]
            curr_labels = labels_train[curr_sample_indices]

            curr_tree = decision_tree.DecisionTree(classes_=self.classes_,
                                                   random_state=None,
                                                   max_features=self.max_features,
                                                   extremely_randomized=self.extremely_randomized,
                                                   labels_encoded=True)
            _t1_debug = time.perf_counter()
            curr_tree.fit(curr_input, curr_labels)
            _t2_debug = time.perf_counter()

            print("Time spent training a single tree: %f..." % (_t2_debug - _t1_debug))

            self.trees.append(curr_tree)

    def predict(self, data):
        # turn probability vectors into a class labels
        preds = np.argmax(self.predict_proba(data), axis=1)
        preds = np.array([self.classes_[idx] for idx in preds])

        return preds

    def predict_proba(self, data):
        num_new_examples = data.shape[0]
        preds = np.zeros((num_new_examples, self.classes_.shape[0]))

        # average the probabilities returned by all trees in forest
        for idx_tree in range(self.num_trees):
            preds += self.trees[idx_tree].predict_proba(data)

        preds = np.divide(preds, self.num_trees)

        return preds