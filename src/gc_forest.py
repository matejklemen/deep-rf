import numpy as np
import random_forest

def train_test_split(features, labels, test_size):
    """
    :param data:
    :param test_size: float between 0 and 1
    :return: (features_train, labels_train, features_test, labels_test)
    """
    return (features[:int((1 - test_size) * features.shape[0]), :], labels[:int((1 - test_size) * features.shape[0])],
            features[int((1 - test_size) * features.shape[0]):, :], labels[int((1 - test_size) * features.shape[0]):])


class GCForest:
    # TODO: add k_cv (for k-fold cv), add val_size (for evaluating cascade after adding new level),
    # max_cascade_layers
    def __init__(self, nrforests_layer=4, ncrforests_layer=4, k_cv=10, label_idx_mapping=None):

        self.nrforests_layer = nrforests_layer
        self.ncrforests_layer = ncrforests_layer

        self.label_idx_mapping = label_idx_mapping
        self.idx_label_mapping = None
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

        self.k_cv = k_cv
        self.val_size = 0.2
        self.max_cascade_depth = 2

        # will be used to store models that make up the whole cascade
        self._cascade = None
        self._num_layers = 0
        self._mg_scan_models = None

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
        # TODO: double check whether decision to add new layer is based on previous layer accuracy or overall accuracy
        best_acc = 0

        while True:
            if self._num_layers >= self.max_cascade_depth:
                print("[fit()] Achieved max allowed depth for gcForest. Exiting...")
                break

            print("[fit()] Training layer %d..." % self._num_layers)
            curr_layer_models, new_features = self._cascade_layer(curr_input, y_train)

            # expand new level
            self._cascade.append(curr_layer_models)
            self._num_layers += 1

            print("[fit()] Shape of new features: ")
            print(new_features.shape)
            # check performance of cascade on validation set
            new_layer_acc = self._eval_cascade(X_val, y_val)
            print("[fit()] New layer accuracy is %.3f..." % new_layer_acc)

            if self._num_layers == 0:
                print("[fit()] Setting first best accuracy to %.3f..." % new_layer_acc)
                best_acc = new_layer_acc

            # drop in accuracy on validation set is bigger than allowed
            if new_layer_acc <= best_acc:
                print("[fit()] New layer accuracy (%.3f) is <= than overall best accuracy (%.3f),"
                      " therefore no more layers will be added to cascade..."
                      % (best_acc, new_layer_acc))
                break
            else:
                print("[fit()] Setting new best accuracy to %.3f..." % new_layer_acc)
                best_acc = new_layer_acc

            curr_input = np.hstack((X_train, new_features))

        print("[fit()] Final verdict: num_layers = %d, best accuracy obtained: %3f..." % (self._num_layers, best_acc))

    def predict(self, X_test):
        return self._predict(X_test, predict_probabilities=False)

    def predict_proba(self, X_test):
        return self._predict(X_test, predict_probabilities=True)

    def mg_scan(self, X, y, window_size=50, stride=1):
        if self.label_idx_mapping is None:
            self._assign_labels(y)

        self._mg_scan_models = []

        slices, labels = self._slice_data(X, y, window_size, stride)

        # completely random forest
        model, feats = self._get_class_distrib(slices, labels, random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping,
                                                                                          extremely_randomized=True))
        self._mg_scan_models.append(model)

        # gather up parts of representation (consecutive rows in 'feats' np.ndarray) for each example
        transformed_feats_crf = np.reshape(feats, [X.shape[0], len(self.label_idx_mapping) * int(feats.shape[0] / X.shape[0])])

        # random forest
        model, feats = self._get_class_distrib(slices, labels, random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping))

        self._mg_scan_models.append(model)

        transformed_feats_rf = np.reshape(feats, [X.shape[0], len(self.label_idx_mapping) * int(feats.shape[0] / X.shape[0])])

        return np.concatenate((transformed_feats_crf, transformed_feats_rf), axis=1)

    def _slice_data(self, X, y, window_size, stride):

        sliced_X = []
        labels = []

        for idx_example in range(X.shape[0]):
            example = X[idx_example, :]
            # print(example)

            for idx in range(0, example.shape[0] - window_size + 1, stride):
                curr_slice = example[idx: idx + window_size]
                print(curr_slice)

                sliced_X.append(curr_slice)
                labels.append(y[idx_example])

        return np.array(sliced_X), np.array(labels)

    def _predict(self, X_test, predict_probabilities=False):
        if self._num_layers <= 0:
            raise Exception("[predict()] Number of layers is <= 0...")

        num_labels = len(self.label_idx_mapping)

        curr_input = np.array(X_test)

        for idx_layer in range(self._num_layers):
            print("[predict()] Going through layer %d..." % idx_layer)
            curr_layer_models = self._cascade[idx_layer]

            new_features = np.zeros((curr_input.shape[0], (self.ncrforests_layer + self.nrforests_layer) * num_labels))
            for idx_model in range(len(curr_layer_models)):
                new_features[:, idx_model * num_labels: (idx_model + 1) * num_labels] += \
                    curr_layer_models[idx_model].predict_proba(curr_input)

            # last layer: get class distributions (normal procedure) and average them to obtain final distribution
            if idx_layer == self._num_layers - 1:
                print("[predict()] Got to the last level...")
                final_probs = np.zeros((curr_input.shape[0], num_labels))
                for idx_model in range(len(curr_layer_models)):
                    final_probs += new_features[:, idx_model * num_labels: (idx_model + 1) * num_labels]

                final_probs = np.divide(final_probs, len(curr_layer_models))

                if predict_probabilities:
                    return final_probs
                # get most probable class
                else:
                    label_indices = np.argmax(final_probs, axis=1)

                    preds = [self.idx_label_mapping[idx] for idx in label_indices]
                    return preds

            # all but the last layer: get the input concatenated with obtained class distribution vectors
            else:
                print("[predict()] I ain't fucking leaving! Concatenating input with new features...")
                curr_input = np.hstack((X_test, new_features))

    def _eval_cascade(self, X_val, y_val):
        """ Evaluates currently built cascade
        :param X_val: validation set features
        :param y_val: validation set labels
        :return: accuracy of cascade
        """
        preds = self._predict(X_val)

        cascade_acc = np.sum(preds == y_val) / y_val.shape[0]
        print("[_eval_cascade()] Evaluated cascade and got accuracy %.3f..." % cascade_acc)

        return cascade_acc

    def _cascade_layer(self, X, y):
        """
        :param X: input data (features)
        :param y: labels
        :return: (list of trained models for current layer, distribution vector for current layer)
        """

        num_labels = len(self.label_idx_mapping)
        curr_layer_models = []
        curr_layer_distributions = np.zeros((X.shape[0], (self.ncrforests_layer + self.nrforests_layer) * num_labels))

        # -- completely random forests --
        for idx_curr_forest in range(self.ncrforests_layer):
            print("Training completely random forest number %d..." % idx_curr_forest)
            # each random forest produces a (#classes)-dimensional vector of class distribution
            rf_obj = random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping,
                                                n_estimators=500,
                                                extremely_randomized=True)
            curr_rf, curr_class_distrib = self._get_class_distrib(X, y, rf_obj)

            curr_layer_models.append(curr_rf)
            curr_layer_distributions[:, idx_curr_forest * num_labels: (idx_curr_forest + 1) * num_labels] += \
                curr_class_distrib

        # -- random forests --
        for idx_curr_forest in range(self.nrforests_layer):
            print("Training random forest number %d..." % idx_curr_forest)
            # each random forest produces a (#classes)-dimensional vector of class distribution
            rf_obj = random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping, n_estimators=500)
            curr_rf, curr_class_distrib = self._get_class_distrib(X, y, rf_obj)

            curr_layer_models.append(curr_rf)
            curr_layer_distributions[:, (self.ncrforests_layer + idx_curr_forest) * num_labels:
                                        (self.ncrforests_layer + idx_curr_forest + 1) * num_labels] += \
                curr_class_distrib

        return curr_layer_models, curr_layer_distributions

    def _get_class_distrib(self, X_train, y_train, model):
        """ Obtains class distribution of a model in a cascade layer.
        :param X_train: training data (features)
        :param y_train: training data (labels)
        :param model:
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

            model.fit(curr_train_X, curr_train_y)
            # careful about vector index - label relationship (certain index should represent same label in all cases)
            class_distrib[curr_test_mask, :] += model.predict_proba(curr_test_X)

        # train a RF model on whole training set, will be placed in cascade
        model.fit(X_train, y_train)

        return model, class_distrib

    def _kfold_cv(self, num_examples, k):
        """ Prepare groups for k-fold cross validation.
        :param num_examples: number of examples in data set
        :param k: number of groups
        :return: np.array of size [1, num_examples] containing group ids ranging from 0 to k - 1.
        """

        if k < num_examples:
            raise Exception("Number of examples is lower than number of groups in k-fold CV...")

        limits = np.linspace(0, num_examples, k, endpoint=False)
        bins = np.digitize(np.arange(0, num_examples), limits) - 1

        return np.random.permutation(bins)