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
    def __init__(self, window_sizes,
                 nrforests_layer=4,
                 ncrforests_layer=4,
                 max_cascade_depth=5,
                 n_estimators=500,
                 val_size=0.2,
                 k_cv=10,
                 label_idx_mapping=None):
        """
        :param window_sizes: an integer or a list of integers, representing different window sizes in multi-grained scanning
        :param nrforests_layer: an integer, determiining the number of random forests in each layer of cascade forest
        :param ncrforests_layer: an integer, determining the number of completely (= extremely) randomized forests
                                in each layer of cascade forest
        :param max_cascade_depth: an integer, determining maximum allowed depth for training cascade forest
        :param n_estimators: number of trees in a random/completely random forest
        :param val_size: a float in the range [0, 1], determining the relative size of validation set that is used
                        during the training of gcForest
        :param k_cv: an integer, determining the number of folds in k-fold cross validation
        :param label_idx_mapping: map from class label to index in probability vector
        """

        self.nrforests_layer = nrforests_layer
        self.ncrforests_layer = ncrforests_layer
        self.window_sizes = [window_sizes] if isinstance(window_sizes, int) else window_sizes

        self.label_idx_mapping = label_idx_mapping
        self.idx_label_mapping = None
        if self.label_idx_mapping is not None:
            self.idx_label_mapping = {self.label_idx_mapping[label]: label for label in self.label_idx_mapping}

        self.k_cv = k_cv
        self.val_size = val_size
        self.max_cascade_depth = max_cascade_depth
        self.n_estimators = n_estimators

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

        # transform input features for each window size
        transformed_features = [self._mg_scan(X_train, y_train, window_size=w_size) for w_size in self.window_sizes]

        # (X_train, y_train, X_val, y_val) for each window size
        split_transformed_features = [train_test_split(feats, y_train, self.val_size) for feats in transformed_features]

        self._cascade = []
        # data split for the first window size
        curr_input_X, curr_input_y, curr_val_X, curr_val_y = split_transformed_features[0]
        prev_acc = 0

        while True:
            if self._num_layers >= self.max_cascade_depth:
                print("[fit()] Achieved max allowed depth for gcForest. Exiting...")
                break

            print("[fit()] Training layer %d..." % self._num_layers)
            curr_layer_models, new_features = self._cascade_layer(curr_input_X, curr_input_y)

            # expand new level
            self._cascade.append(curr_layer_models)
            self._num_layers += 1

            print("[fit()] Shape of new features: ")
            print(new_features.shape)

            # extract validation sets for each window size
            transformed_val_X = [quad[2] for quad in split_transformed_features]

            # check performance of cascade on validation set
            new_layer_acc = self._eval_cascade(transformed_val_X, curr_val_y)
            print("[fit()] New layer accuracy is %.3f..." % new_layer_acc)

            # if accuracy (with new layer) on validation set does not increase, remove the new layer and quit training
            if new_layer_acc <= prev_acc:
                print("[fit()] New layer accuracy (%.3f) is <= than overall best accuracy (%.3f),"
                      " therefore no more layers will be added to cascade..."
                      % (prev_acc, new_layer_acc))

                del self._cascade[-1]
                self._num_layers -= 1

                break

            print("[fit()] Setting new best accuracy to %.3f..." % new_layer_acc)
            prev_acc = new_layer_acc

            print("Picking up data for %d..." % (self._num_layers % len(self.window_sizes)))
            raw_curr_input_X, curr_input_y, curr_val_X, curr_val_y = split_transformed_features[self._num_layers %
                                                                                                len(self.window_sizes)]

            curr_input_X = np.hstack((raw_curr_input_X, new_features))

        print("[fit()] Final verdict: num_layers = %d, best accuracy obtained: %3f..." % (self._num_layers, prev_acc))

    def predict(self, X_test):
        transformed_features = [self._mg_scan(X_test, window_size=w_size) for w_size in self.window_sizes]
        return self._predict(transformed_features, predict_probabilities=False)

    def predict_proba(self, X_test):
        transformed_features = [self._mg_scan(X_test, window_size=w_size) for w_size in self.window_sizes]
        return self._predict(transformed_features, predict_probabilities=True)

    def _mg_scan(self, X, y=None, window_size=50, stride=1):
        if self.label_idx_mapping is None:
            self._assign_labels(y)

        slices, labels = self._slice_data(X, y, window_size, stride)

        # train models on obtained slices
        if y is not None:
            self._mg_scan_models = []

            # completely random forest
            model_crf, feats_crf = self._get_class_distrib(slices, labels, random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping,
                                                                                                      n_estimators=self.n_estimators,
                                                                                                      extremely_randomized=True))
            # random forest
            model_rf, feats_rf = self._get_class_distrib(slices, labels, random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping,
                                                                                                    n_estimators=self.n_estimators))

            self._mg_scan_models.append(model_crf)
            self._mg_scan_models.append(model_rf)
        else:
            # TODO: make this more general as it currently would not work for multiple window sizes
            crf_model = self._mg_scan_models[0]
            rf_model = self._mg_scan_models[1]

            feats_crf = crf_model.predict_proba(slices)
            feats_rf = rf_model.predict_proba(slices)

        # gather up parts of representation (consecutive rows in feats np.ndarray) for each example
        transformed_feats_crf = np.reshape(feats_crf, [X.shape[0], len(self.label_idx_mapping) * int(feats_crf.shape[0] / X.shape[0])])
        transformed_feats_rf = np.reshape(feats_rf, [X.shape[0], len(self.label_idx_mapping) * int(feats_rf.shape[0] / X.shape[0])])

        return np.concatenate((transformed_feats_crf, transformed_feats_rf), axis=1)

    def _slice_data(self, X, y, window_size, stride):

        sliced_X = []
        labels = []

        for idx_example in range(X.shape[0]):
            example = X[idx_example, :]
            # print(example)

            for idx in range(0, example.shape[0] - window_size + 1, stride):
                curr_slice = example[idx: idx + window_size]

                sliced_X.append(curr_slice)
                if y is not None:
                    labels.append(y[idx_example])

        features = np.array(sliced_X)
        labels = np.array(labels) if y is not None else None

        return features, labels

    def _predict(self, transformed_features, predict_probabilities=False):
        # X_test ... list of transformed feature arrays (a new feature array for each window size)
        if self._num_layers <= 0:
            raise Exception("[predict()] Number of layers is <= 0...")

        num_labels = len(self.label_idx_mapping)
        curr_input = transformed_features[0]

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
                print("Created a vector for final predictions of shape...")
                print(final_probs.shape)

                for idx_model in range(len(curr_layer_models)):
                    final_probs += new_features[:, idx_model * num_labels: (idx_model + 1) * num_labels]

                final_probs = np.divide(final_probs, len(curr_layer_models))

                if predict_probabilities:
                    return final_probs
                # get most probable class
                else:
                    label_indices = np.argmax(final_probs, axis=1)
                    print("Vector of label indices has a shape of...")
                    print(label_indices.shape)
                    preds = [self.idx_label_mapping[idx] for idx in label_indices]
                    return np.array(preds)

            # all but the last layer: get the input concatenated with obtained class distribution vectors
            else:
                print("[predict()] I ain't fucking leaving! Concatenating input with new features...")
                curr_input = np.hstack((transformed_features[(idx_layer + 1) % len(self.window_sizes)], new_features))

    def _eval_cascade(self, X_val, y_val):
        """ Evaluates currently built cascade
        :param X_val: list of validation set transformed features (possibly obtained with multiple sliding window sizes)
        :param y_val: validation set labels
        :return: accuracy of cascade
        """

        print("[_eval_cascade()] Evaluating cascade on validation data of len %d ( = number of different window sizes)" % len(X_val))

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
                                                n_estimators=self.n_estimators,
                                                extremely_randomized=True)
            curr_rf, curr_class_distrib = self._get_class_distrib(X, y, rf_obj)

            curr_layer_models.append(curr_rf)
            curr_layer_distributions[:, idx_curr_forest * num_labels: (idx_curr_forest + 1) * num_labels] += \
                curr_class_distrib

        # -- random forests --
        for idx_curr_forest in range(self.nrforests_layer):
            print("Training random forest number %d..." % idx_curr_forest)
            # each random forest produces a (#classes)-dimensional vector of class distribution
            rf_obj = random_forest.RandomForest(label_idx_mapping=self.label_idx_mapping, n_estimators=self.n_estimators)
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

        if num_examples < k:
            raise Exception("Number of examples (num_examples=%d) is lower than number of groups in k-fold CV (k=%d)..."
                            % (num_examples, k))

        limits = np.linspace(0, num_examples, k, endpoint=False)
        bins = np.digitize(np.arange(0, num_examples), limits) - 1

        return np.random.permutation(bins)