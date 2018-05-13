import numpy as np
import random_forest

def train_test_split(features, labels, test_size):
    """
    :param data:
    :param test_size: float between 0 and 1
    :return: (features_train, labels_train, features_test, labels_test)
    """
    return (features[:int(test_size * features.shape[0]), :], labels[:int(test_size * features.shape[0])],
            features[int(test_size * features.shape[0]):, :], labels[int(test_size) * features.shape[0]:])


class GCForest:
    # TODO: add k_cv (for k-fold cv), add val_size (for evaluating cascade after adding new level),
    # max_cascade_layers
    def __init__(self, nforests_cascade=4, label_idx_mapping=None):

        self.nforests_cascade = nforests_cascade

        self.label_idx_mapping = label_idx_mapping
        self.idx_label_mapping = None
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

        self.k_cv = 10
        self.val_size = 0.2
        self.max_cascade_depth = 2

        # will be used to store models that make up the whole cascade
        self._cascade = None
        self._num_layers = 0

    def _assign_labels(self, labels_train):
        # get unique labels and map them to indices of output (probability) vector
        unique_labels = set(labels_train)
        self.label_idx_mapping = dict(zip(unique_labels, range(len(unique_labels))))
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

    def fit(self, X_train, y_train):
        if self.label_idx_mapping is None:
            self._assign_labels(y_train)

        X_train, y_train, X_val, y_val = train_test_split(X_train, y_train, self.val_size)

        idx_curr_layer = 0
        self._cascade = []
        curr_input = X_train

        while True:
            if idx_curr_layer >= self.max_cascade_depth:
                print("Achieved max allowed depth for gcForest. Exiting...")
                break

            print("Training layer %d..." % idx_curr_layer)
            curr_layer_models, new_features = self._cascade_layer()

            print("Shape of new features: ")
            print(new_features.shape)
            # TODO: check performance of cascade on validation set
            # ...

            # if performance keeps on improving, store current layer
            self._cascade.append(curr_layer_models)
            idx_curr_layer += 1
            self._num_layers = idx_curr_layer
            # else
            # stop building new layers, average class probabilities in new_features and select class with max probability

            curr_input = np.hstack((X_train, new_features))
            self._num_layers = idx_curr_layer

    # TODO: test this (function is untested
    def predict(self, X_test, predict_probabilities=False):
        num_labels = len(self.label_idx_mapping)

        curr_input = np.array(X_test)

        for idx_layer in range(self._num_layers):
            curr_layer_models = self._cascade[idx_layer]

            new_features = np.zeros(curr_input.shape[0], self.nforests_cascade * num_labels)
            for idx_model in range(len(curr_layer_models)):
                new_features[:, idx_model * num_labels: (idx_model + 1) * num_labels] += \
                    curr_layer_models[idx_model].predict(curr_input, return_probabilities=True)

            # last layer: get class distributions (normal procedure) and average them to obtain final distribution
            if idx_layer == self._num_layers - 1:
                final_probs = np.zeros((curr_input.shape[0], num_labels))
                for idx_model in range(self.nforests_cascade):
                    final_probs += new_features[:, idx_model * num_labels: (idx_model + 1) * num_labels]

                final_probs = np.divide(final_probs, self.nforests_cascade)

                if predict_probabilities:
                    return final_probs
                # get most probable class
                else:
                    label_indices = np.argmax(final_probs, axis=1)

                    preds = [self.idx_label_mapping[idx] for idx in label_indices]
                    return preds

            # all but the last layer: get the input concatenated with obtained class distribution vectors
            else:
                curr_input = np.hstack((X_test, new_features))

    def _eval_cascade(self):
        pass

    def _cascade_layer(self, X, y):
        """
        :param X: input data (features)
        :param y: labels
        :return:
        """

        num_labels = len(self.label_idx_mapping)
        curr_layer_models = []
        curr_layer_distributions = np.zeros(X.shape[0], self.nforests_cascade * num_labels)

        for idx_curr_forest in range(self.nforests_cascade):
            # TODO add completely random forests
            # each random forest produces a (#classes)-dimensional vector of class distribution
            curr_rf, curr_class_distrib = self._get_class_distrib(X, y)

            curr_layer_models.append(curr_rf)
            curr_layer_distributions[:, idx_curr_forest * num_labels: (idx_curr_forest + 1) * num_labels] += \
                curr_class_distrib

        return (curr_layer_models, curr_layer_distributions)

    def _get_class_distrib(self, X_train, y_train):
        """ Obtains class distribution of a single random forest in a cascade layer.
        :param X_train: training data (features)
        :param y_train: training data (labels)
        :return: tuple, consisting of (random_forest.RandomForest model, class distribution) where
                class distribution has same number of rows as X_train and (#labels) columns
        """

        bins = self._kfold_cv(X_train.shape[0], self.k_cv)
        class_distrib = np.zeros((X_train.shape[0], len(self.label_idx_mapping)))

        # k-fold cross validation to obtain class distribution
        for idx_test_bin in range(self.k_cv):
            curr_test_mask = (bins == idx_test_bin)
            curr_train_X, curr_train_y = X_train[np.logical_not(curr_test_mask), :], y_train[np.logical_not(curr_test_mask)]
            curr_test_X, curr_test_y = X_train[curr_test_mask, :], y_train[curr_test_mask]

            rf = random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping)
            rf.fit(curr_train_X, curr_train_y)

            class_distrib[curr_test_mask, :] += rf.predict(curr_test_X, return_probabilities=True)

        # train a RF model on whole training set, will be placed in cascade
        rf = random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping)
        rf.fit(X_train, y_train)

        return (rf, class_distrib)

    def _kfold_cv(self, num_examples, k):
        """ Prepare groups for k-fold cross validation.
        :param num_examples: number of examples in data set
        :param k: number of groups
        :return: np.array of size [1, num_examples] containing group ids ranging from 0 to k - 1.
        """
        limits = np.linspace(0, num_examples, k, endpoint=False)
        bins = np.digitize(np.arange(0, num_examples), limits) - 1

        return np.random.permutation(bins)

    def predict(self):
        pass

