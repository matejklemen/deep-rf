import numpy as np

from mg_scanning import Grain, MultiGrainedScanning
from cascade_forest import CascadeLayer, CascadeForest, EndingLayerAverage, EndingLayerStacking


class GrainedCascadeForest:
    def __init__(self, single_shape=None,
                 n_rf_grain=1,
                 n_crf_grain=1,
                 n_rsf_grain=0,
                 n_xonf_grain=0,
                 n_rf_cascade=2,
                 n_crf_cascade=2,
                 n_rsf_cascade=0,
                 n_xonf_cascade=0,
                 end_layer_cascade="avg",
                 window_sizes=None,
                 strides=None,
                 n_estimators=100,
                 n_estimators_xonf=100,
                 k_cv=3,
                 val_size=0.2,
                 early_stop_iters=1,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False):

        # multi-grained scanning parameters
        self.n_rf_grain = n_rf_grain
        self.n_crf_grain = n_crf_grain
        self.n_rsf_grain = n_rsf_grain
        self.n_xonf_grain = n_xonf_grain
        self.single_shape = single_shape
        self.window_sizes = window_sizes if window_sizes is not None else None
        self.strides = strides if strides is not None else None

        # cascade forest parameters
        self.n_rf_cascade = n_rf_cascade
        self.n_crf_cascade = n_crf_cascade
        self.n_rsf_cascade = n_rsf_cascade
        self.n_xonf_cascade = n_xonf_cascade
        self.end_layer_cascade = end_layer_cascade

        # general parameters
        self.n_estimators = n_estimators
        self.n_estimators_xonf = n_estimators_xonf
        self.k_cv = k_cv
        self.val_size = val_size
        self.early_stop_iters = early_stop_iters
        self.classes_ = classes_
        if random_state is not None:
            np.random.seed(random_state)
        self.labels_encoded = labels_encoded

        # TODO: add parameters
        self.cache_dir = "tmp/"

        # miscellaneous
        self._grains = []
        self._mgscan = None
        self._casc_forest = None

    def _assign_labels(self, labels_train):
        if self.classes_ is None:
            # wanted classes not provided
            self.classes_, encoded_labels = np.unique(labels_train, return_inverse=True)
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
                               n_rsf=self.n_rsf_grain,
                               n_xonf=self.n_xonf_grain,
                               n_estimators=self.n_estimators,
                               n_estimators_xonf=self.n_estimators_xonf,  # TODO
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

        self._prepare_grains()
        mg_scan = MultiGrainedScanning(grains=self._grains) if len(self._grains) > 0 else None

        # features that will be used in cascade forest - if multi-grained scanning was not requested,
        # use only raw features
        transformed_feats = mg_scan.train_all_grains(feats=feats, labels=labels) if mg_scan is not None else [feats]
        print("[fit(...)] Multi-grained scanning shapes...")
        for feats in transformed_feats:
            print("[fit(...)] -> %s" % str(feats.shape))

        prev_acc, curr_acc = -1, 0
        idx_curr_layer = 0
        num_opt_layers = 0

        # curr_input, curr_labels = train_transformed_X[0], train_transformed_y[0]
        curr_input, curr_labels = transformed_feats[0], labels
        # TODO: add options for switching models in last layer
        cascade_forest = CascadeForest(classes_=self.classes_, ending_layer=self.end_layer_cascade, k_cv=self.k_cv)

        while True:
            print("[fit(...)] Adding cascade layer %d..." % idx_curr_layer)
            cascade_forest.add_layer(CascadeLayer(n_rf=self.n_rf_cascade,
                                                  n_crf=self.n_crf_cascade,
                                                  n_rsf=self.n_rsf_cascade,
                                                  n_xonf=self.n_xonf_cascade,
                                                  n_estimators=self.n_estimators,
                                                  n_estimators_xonf=self.n_estimators_xonf,
                                                  k_cv=self.k_cv,
                                                  classes_=self.classes_,
                                                  labels_encoded=True,
                                                  keep_models=False))

            curr_feats = cascade_forest.train_next_layer(feats=curr_input, labels=curr_labels)

            # k-fold cross-validation accuracy to determine optimal number of layers
            curr_acc = cascade_forest.layers[-1].kfold_acc

            curr_input = np.hstack((transformed_feats[idx_curr_layer % len(transformed_feats)], curr_feats))

            if curr_acc <= prev_acc:
                print("[fit(...)] Current accuracy <= previous accuracy... (%.5f <= %.5f)" %
                      (curr_acc, prev_acc))
            else:
                print("[fit(...)] Current accuracy > previous accuracy... (%.5f > %.5f)" % (curr_acc, prev_acc))
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
        self._casc_forest = CascadeForest(classes_=self.classes_, ending_layer=self.end_layer_cascade, k_cv=self.k_cv)

        # retrain using entire data set
        curr_input = transformed_feats[0]

        # (num_opt_layers + 1) because num_opt_layers holds index of last useful layer (0-based)
        for idx_layer in range(num_opt_layers + 1):
            print("[fit(...)] Retraining layer %d..." % idx_layer)
            self._casc_forest.add_layer(CascadeLayer(n_rf=self.n_rf_cascade,
                                                     n_crf=self.n_crf_cascade,
                                                     n_rsf=self.n_rsf_cascade,
                                                     n_xonf=self.n_xonf_cascade,
                                                     n_estimators=self.n_estimators,
                                                     n_estimators_xonf=self.n_estimators_xonf,
                                                     k_cv=self.k_cv,
                                                     classes_=self.classes_,
                                                     labels_encoded=True))

            curr_feats = self._casc_forest.train_next_layer(feats=curr_input, labels=labels)
            print("[fit(...)] Concatenating features of layer %d with new feats..." % (idx_layer % len(transformed_feats)))
            curr_input = np.hstack((transformed_feats[idx_layer % len(transformed_feats)], curr_feats))

        self._casc_forest.ending_layer.fit(curr_input, labels)

        print("[fit(...)] Done training!\n")

    # simultaneously fit layers on training data and predict for new data (using trained layer)
    # done in an attempt to try to avoid having to save all models to disk and then re-loading them and predicting
    def fit_predict(self, train_feats, train_labels, test_feats):
        if not self.labels_encoded:
            train_labels = self._assign_labels(train_labels)

        self._prepare_grains()
        mg_scan = MultiGrainedScanning(grains=self._grains) if len(self._grains) > 0 else None

        # features that will be used in cascade forest - if multi-grained scanning was not requested,
        # use only raw features
        if mg_scan is not None:
            print("[fit_predict(...)] Performing multi-grained scanning...")
            train_transformed_feats, test_transformed_feats = mg_scan.fit_transform_all_grains(train_feats=train_feats,
                                                                                               train_labels=train_labels,
                                                                                               test_feats=test_feats)
        else:
            print("[fit_predict(...)] Multi-grained scanning was not requested so defaulting to raw features...")
            train_transformed_feats, test_transformed_feats = [train_feats], [test_feats]

        print("[fit_predict(...)] Multi-grained scanning shapes...")
        for feats in train_transformed_feats:
            print("[fit_predict(...)] -> %s" % str(feats.shape))

        if self.end_layer_cascade == "avg":
            end_layer = EndingLayerAverage(classes_=self.classes_)
        elif self.end_layer_cascade == "stack":
            end_layer = EndingLayerStacking(classes_=self.classes_, k_cv=self.k_cv)
        else:
            raise NotImplementedError("'ending_layer' must be one of {%s}" % ",".join(["avg", "stack"]))

        prev_acc, curr_acc = -1, 0
        idx_curr_layer = 0
        num_opt_layers = 0

        curr_train_input, curr_train_labels = train_transformed_feats[0], train_labels
        curr_test_input = test_transformed_feats[0]

        while True:
            curr_layer = CascadeLayer(n_rf=self.n_rf_cascade,
                                      n_crf=self.n_crf_cascade,
                                      n_rsf=self.n_rsf_cascade,
                                      n_xonf=self.n_xonf_cascade,
                                      n_estimators=self.n_estimators,
                                      n_estimators_xonf=self.n_estimators_xonf,
                                      k_cv=self.k_cv,
                                      classes_=self.classes_,
                                      labels_encoded=True,
                                      keep_models=False)

            curr_train_feats, curr_test_feats = curr_layer.fit_transform(train_feats=curr_train_input,
                                                                         train_labels=curr_train_labels,
                                                                         test_feats=curr_test_input)

            # k-fold cross-validation accuracy to determine optimal number of layers
            curr_acc = curr_layer.kfold_acc

            curr_train_input = np.hstack((train_transformed_feats[idx_curr_layer % len(train_transformed_feats)],
                                          curr_train_feats))
            curr_test_input = np.hstack((test_transformed_feats[idx_curr_layer % len(test_transformed_feats)],
                                         curr_test_feats))

            if curr_acc <= prev_acc:
                print("[fit_predict(...)] Current accuracy <= previous accuracy... (%.5f <= %.5f)" %
                      (curr_acc, prev_acc))
            else:
                print("[fit_predict(...)] Current accuracy > previous accuracy... (%.5f > %.5f)" % (curr_acc, prev_acc))
                # act as if every layer with higher accuracy is the last layer
                preds = end_layer.fit_predict(train_feats=curr_train_feats,
                                              train_labels=curr_train_labels,
                                              test_feats=curr_test_feats)

                prev_acc = curr_acc
                num_opt_layers = idx_curr_layer

            # early stopping: if the accuracy doesn't improve for 'early_stop_iters' in a row, finish the process
            if idx_curr_layer - num_opt_layers == self.early_stop_iters:
                print("[fit_predict(...)] Accuracy has not increased for %d rounds in a row..." % self.early_stop_iters)
                break

            idx_curr_layer += 1

        return preds

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
