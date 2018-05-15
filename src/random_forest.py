import decision_tree
import numpy as np


class RandomForest:
    def __init__(self, n_estimators=100,
                 max_depth=10,
                 attr_types=None,
                 label_idx_mapping=None,
                 random_state=None,
                 max_features=None,
                 extremely_randomized=False):
        """
        :param num_trees:
        :param max_depth:
        :param label_idx_mapping: map from class label to index in probability vector
        :param attr_types: an iterable representing attribute types that will be passed into fit() (and predict()).
        Most likely not needed, but the heuristic for determining attribute types may sometimes fail (if very raw
        data is passed in).
        :param random_state: an integer determining the random state for random number generator.
        :param extremely_randomized: a boolean determining whether to build a completely random forest

        (not a class param) idx_label_mapping: map from index in probability vector to class label
        Example:
            label_idx_mapping = {'C1': 2, 'C2': 1, 'C3': 0}
            idx_label_mapping = {1: 'C2', 0: 'C3', 2: 'C1'}
            Probability of class 'C1' is stored on the index 2 in probability vector,
            'C2' on the index 1 and 'C3' on the index 2.
        """
        self.num_trees = n_estimators
        self.max_depth = max_depth
        self.trees = []

        self.label_idx_mapping = label_idx_mapping
        self.idx_label_mapping = None
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

        self.attr_types = attr_types
        self.random_state = random_state
        if self.random_state:
            np.random.seed(self.random_state)

        self.max_features = max_features
        self.extremely_randomized = extremely_randomized

    def _assign_labels(self, labels_train):
        # get unique labels and map them to indices of output (probability) vector
        unique_labels = set(labels_train)
        self.label_idx_mapping = dict(zip(unique_labels, range(len(unique_labels))))
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

    def fit(self, input_train, labels_train):
        if self.label_idx_mapping is None:
            self._assign_labels(labels_train)

        self.trees = []
        sample_size = np.size(input_train, axis=0)
        row_indices = np.array(list(range(sample_size)))

        # if number of used features is not specified, default to sqrt(number of all features)
        if self.max_features is None:
            self.max_features = int(np.sqrt(np.size(input_train, axis=1)))

        for idx_tree in range(self.num_trees):

            # choose sample for current tree (random with replacement)
            curr_sample_indices = np.random.choice(row_indices, sample_size)

            curr_input = input_train[curr_sample_indices, :]
            curr_labels = labels_train[curr_sample_indices]

            curr_tree = decision_tree.DecisionTree(label_idx_mapping=self.label_idx_mapping,
                                                   random_state=self.random_state,
                                                   max_features=self.max_features,
                                                   extremely_randomized=self.extremely_randomized)
            curr_tree.fit(curr_input, curr_labels)

            self.trees.append(curr_tree)

    def predict(self, data):
        # turn probability vectors into a class labels
        preds = np.argmax(self.predict_proba(data), axis=1)
        preds = np.array([self.idx_label_mapping[idx] for idx in preds])

        return preds

    def predict_proba(self, data):
        num_new_examples = data.shape[0]
        preds = np.zeros((num_new_examples, len(self.label_idx_mapping)))

        # average the probabilities returned by all trees in forest
        for idx_tree in range(self.num_trees):
            preds += self.trees[idx_tree].predict_proba(data)

        preds = np.divide(preds, self.num_trees)

        return preds