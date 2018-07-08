import numpy as np
from mg_scanning import *
from cascade_forest import *
from sklearn.model_selection import train_test_split


# debug
import time

# finds (and returns) where elements of array 'first' are in array 'second'
# warning: very expensive in terms of memory!
# found at: https://stackoverflow.com/a/40449296
def find_reordering(first, second):
    return np.where(second[:, None] == first[None, :])[0]


class GrainedCascadeForest:
    def __init__(self, single_shape=None,
                 n_rf_grain=1,
                 n_crf_grain=1,
                 n_rf_cascade=2,
                 n_crf_cascade=2,
                 window_sizes=None,
                 strides=None,
                 n_estimators=100,
                 k_cv=3,
                 val_size=0.2,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False):

        # multi-grained scanning parameters
        self.n_rf_grain = n_rf_grain
        self.n_crf_grain = n_crf_grain
        self.single_shape = single_shape
        self.window_sizes = window_sizes if window_sizes is not None else None
        self.strides = strides if strides is not None else None

        # cascade forest parameters
        self.n_rf_cascade = n_rf_cascade
        self.n_crf_cascade = n_crf_cascade

        # general parameters
        self.n_estimators = n_estimators
        self.k_cv = k_cv
        self.val_size = val_size
        self.classes_ = classes_
        if random_state is not None:
            np.random.seed(random_state)
        self.labels_encoded = labels_encoded

        # TODO: add parameters
        self.early_stop_val = True
        self.early_stop_iters = 4

        # miscellaneous
        self._grains = []
        self._mgscan = None
        self._casc_forest = None

    def _assign_labels(self, labels_train):
        if self.classes_ is None:
            # wanted classes not provided
            self.classes_, encoded_labels = np.unique(labels_train, return_inverse=True)
            # TODO: self.labels_encoded = True?
        else:
            encoded_labels = np.zeros_like(labels_train, np.int32)
            for encoded_label in range(self.classes_.shape[0]):
                encoded_labels[labels_train == self.classes_[encoded_label]] = encoded_label

        return encoded_labels

    def _prepare_grains(self):
        # TODO: check if 'window_sizes' and 'strides' is a list/numpy.ndarray/int/tuple
        # ...

        self._grains = []
        if self.window_sizes is None:
            return

        for idx_grain in range(len(self.window_sizes)):
            curr_grain = Grain(window_size=self.window_sizes[idx_grain],
                               single_shape=self.single_shape,
                               n_crf=self.n_crf_grain,
                               n_rf=self.n_rf_grain,
                               stride=self.strides[idx_grain],
                               k_cv=self.k_cv,
                               classes_=self.classes_,
                               random_state=None,
                               labels_encoded=True)

            self._grains.append(curr_grain)

    def fit(self, feats, labels):
        print("[fit(...)] TRAINING...")
        if not self.labels_encoded:
            labels = self._assign_labels(labels)

        # TODO: train
        self._prepare_grains()
        mg_scan = MultiGrainedScanning(grains=self._grains) if len(self._grains) > 0 else None

        # features that will be used in cascade forest - if multi-grained scanning was not requested,
        # use only raw features
        transformed_feats = mg_scan.train_all_grains(feats=feats, labels=labels) if mg_scan is not None else [feats]
        print("[fit(...)] Multi-grained scanning shapes...")
        for feats in transformed_feats:
            print("[fit(...)] -> %s" % str(feats.shape))

        # train_transformed_X, train_transformed_y = [], []
        # val_transformed_X, val_transformed_y = [], []
        # train_idx, test_idx = train_test_split(range(labels.shape[0]), test_size=self.val_size)
        # for curr_feats in transformed_feats:
        #     # Note: putting features and labels in 4 different arrays in hopes of cleaner code later on
        #     # (reusing cascade_forest.CascadeForest's _pred_proba(...) function)
        #
        #     # train_X, train_y, val_X, val_y = common_utils.train_test_split(feats=curr_feats,
        #     #                                                                labels=labels,
        #     #                                                                test_size=self.val_size)
        #
        #     train_transformed_X.append(curr_feats[train_idx, :])
        #     train_transformed_y.append(labels[train_idx])
        #     val_transformed_X.append(curr_feats[test_idx, :])
        #     val_transformed_y.append(labels[test_idx])

        prev_acc, curr_acc = -1, 0
        idx_curr_layer = 0
        num_opt_layers = 0

        # curr_input, curr_labels = train_transformed_X[0], train_transformed_y[0]
        curr_input, curr_labels = transformed_feats[0], labels
        cascade_forest = CascadeForest(classes_=self.classes_)

        while True:
            print("[fit(...)] Adding cascade layer %d..." % idx_curr_layer)
            cascade_forest.add_layer(CascadeLayer(n_rf=self.n_rf_cascade,
                                                  n_crf=self.n_crf_cascade,
                                                  n_estimators=self.n_estimators,
                                                  k_cv=self.k_cv,
                                                  classes_=self.classes_,
                                                  labels_encoded=True))

            curr_feats = cascade_forest.train_next_layer(feats=curr_input, labels=curr_labels)

            # TODO: remove this bullshit as stopping based on training accuracy is complete bullshit
            # train_preds = np.argmax(cascade_forest.ending_layer.predict_proba(curr_feats), axis=1)
            # train_acc = np.sum(train_preds == train_transformed_y[0]) / train_preds.shape[0]
            # print("[fit(...)] Current TRAINING accuracy is %.5f..." % train_acc)

            # TODO: validate performance
            # val_preds = np.argmax(cascade_forest._pred_proba(val_transformed_X), axis=1)
            # val_acc = np.sum(val_preds == val_transformed_y[0]) / val_preds.shape[0]
            # print("[fit(...)] Current VALIDATION accuracy is %.5f..." % val_acc)

            # curr_acc = val_acc if self.early_stop_val else train_acc
            curr_acc = cascade_forest.layers[-1].kfold_acc

            # TODO: if better accuracy -> set accuracies + new feats, else remove last layer and break out of this loop
            if curr_acc <= prev_acc:
                print("[fit(...)] Current training (kfold) accuracy <= previous accuracy... (%.5f <= %.5f)" %
                      (curr_acc, prev_acc))
            else:
                print("[fit(...)] Current training (kfold) accuracy is higher than previous accuracy... (%.5f > %.5f)" % (curr_acc, prev_acc))
                print("[fit(...)] Concatenating features of layer %d with new feats..." % (idx_curr_layer % len(transformed_feats)))
                curr_input = np.hstack((transformed_feats[idx_curr_layer % len(transformed_feats)], curr_feats))
                prev_acc = curr_acc
                num_opt_layers = idx_curr_layer

            # early stopping: if the accuracy (validation if early_stop_val=True or training if early_stop_val=False)
            # doesn't improve for early_stop_iters in a row, stop trying to grow cascade forest
            if idx_curr_layer - num_opt_layers == self.early_stop_iters:
                print("[fit(...)] Accuracy has not increased for %d rounds in a row..." % self.early_stop_iters)
                break

            idx_curr_layer += 1

        print("[fit(...)] Number of optimal layers was determined to be %d..." % (num_opt_layers + 1))
        del cascade_forest

        self._mgscan = mg_scan
        self._casc_forest = CascadeForest(classes_=self.classes_)

        # retrain using entire data set
        curr_input = transformed_feats[0]
        # TODO: retrain
        # (num_opt_layers + 1) because num_opt_layers holds index of last useful layer (0-based)
        for idx_layer in range(num_opt_layers + 1):
            print("[fit(...)] Retraining layer %d..." % idx_layer)
            self._casc_forest.add_layer(CascadeLayer(n_rf=self.n_rf_cascade,
                                                     n_crf=self.n_crf_cascade,
                                                     n_estimators=self.n_estimators,
                                                     k_cv=self.k_cv,
                                                     classes_=self.classes_,
                                                     labels_encoded=True))

            curr_feats = self._casc_forest.train_next_layer(feats=curr_input, labels=labels)
            print("[fit(...)] Concatenating features of layer %d with new feats..." % (idx_layer % len(transformed_feats)))
            curr_input = np.hstack((transformed_feats[idx_layer % len(transformed_feats)], curr_feats))

        print("[fit(...)] Done training!\n")

    def predict_proba(self, feats):
        print("[predict_proba(...)] Predicting probabilities...")
        if self._casc_forest is None:
            raise Exception("GrainedCascadeForest is not trained yet!")

        transformed_feats = self._mgscan.transform_all_grains(feats=feats) if self._mgscan is not None else [feats]
        print("[predict_proba(...)] Multi-grained scanning shapes...")
        for feats in transformed_feats:
            print("[predict_proba(...)] -> %s" % str(feats.shape))

        return self._casc_forest._pred_proba(transformed_feats)

    def predict(self, feats):
        return self.classes_[np.argmax(self.predict_proba(feats=feats), axis=1)]
