import decision_tree
import numpy as np


class RandomForest:
    def __init__(self, num_trees=100, max_depth=10,
                 label_idx_mapping=None, attr_types=None):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []

        """
            label_idx_mapping... map from class label to index in probability vector        
            idx_label_mapping... map from index in probability vector to class label
            Example: 
                label_idx_mapping = {'C1': 2, 'C2': 1, 'C3': 0} 
                idx_label_mapping = {1: 'C2', 0: 'C3', 2: 'C1'}
                Probability of class 'C1' is stored on the index 2 in probability vector,
                'C2' on the index 1 and 'C3' on the index 2.
        """
        self.label_idx_mapping = label_idx_mapping
        self.idx_label_mapping = None
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

        self.attr_types = attr_types

    def _assign_labels(self, labels_train):
        # get unique labels and map them to indices of output (probability) vector
        unique_labels = set(labels_train)
        self.label_idx_mapping = dict(zip(unique_labels, range(len(unique_labels))))
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

    def fit(self, input_train, labels_train, num_features=None):
        if self.label_idx_mapping is None:
            self._assign_labels(labels_train)

        sample_size = np.size(input_train, axis=0)
        row_indices = np.array(list(range(sample_size)))

        # if number of used features is not specified, default to sqrt(number of all features)
        if num_features is None:
            num_features = int(np.sqrt(np.size(input_train, axis=1)))

        for idx_tree in range(self.num_trees):

            # choose sample for current tree (random with replacement)
            curr_sample_indices = np.random.choice(row_indices, sample_size)

            curr_input = input_train[curr_sample_indices, :]
            curr_labels = labels_train[curr_sample_indices]

            curr_tree = decision_tree.DecisionTree()
            curr_tree.fit(curr_input, curr_labels, num_features)

            self.trees.append(curr_tree)

    def predict(self, data, return_probabilities=False):
        num_new_examples = np.size(data, axis=0)
        preds = np.zeros((num_new_examples, len(self.label_idx_mapping)))

        # average the probabilities returned by all trees in forest
        for idx_tree in range(self.num_trees):
            preds += self.trees[idx_tree].predict(data, return_probabilities=True)

        preds = np.divide(preds, self.num_trees)

        if return_probabilities:
            return preds

        # turn probability vectors into a class labels
        preds = np.argmax(preds, axis=1)
        preds = np.array([self.idx_label_mapping[idx] for idx in preds])

        return preds