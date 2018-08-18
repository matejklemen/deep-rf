import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from gcforest.random_subspace import RandomSubspaceForest
from gcforest.xofn import RandomXOfNForest
from gcforest import common_utils


class CascadeForest:
    def __init__(self, classes_=None, ending_layer="avg", model=None, k_cv=3):
        self.classes_ = np.array(classes_) if classes_ is not None else None
        self.layers = []

        self.idx_fit_next = 0

        options = ["avg", "stack"]

        # this branching is also present in fit_predict() method of GrainedCascadeForest ->
        # TODO: move construction to a single class
        if ending_layer == options[0]:
            self.ending_layer = EndingLayerAverage(classes_=self.classes_)
        elif ending_layer == options[1]:
            self.ending_layer = EndingLayerStacking(classes_=self.classes_, model=model, k_cv=k_cv)
        else:
            raise NotImplementedError("'ending_layer' must be one of {%s}" % ",".join(options))

    def add_layer(self, layer):
        if not isinstance(layer, CascadeLayer):
            raise Exception("'layer' must be an object of type CascadeLayer!")

        self.layers.append(layer)

    def remove_last_layer(self):
        num_layers = len(self.layers)
        if num_layers == 0:
            raise Exception("There are no layers in this CascadeForest!")

        if num_layers == self.idx_fit_next:
            self.idx_fit_next -= 1

        return self.layers.pop()

    def train_next_layer(self, feats, labels):
        num_layers = len(self.layers)
        if num_layers == 0:
            raise Exception("There are no layers in this CascadeForest!")

        if self.idx_fit_next == num_layers:
            raise Exception("Tried to train next layer when they are all already trained!")

        transformed_feats = self.layers[self.idx_fit_next].train_layer(feats, labels)
        self.idx_fit_next += 1

        return transformed_feats

    def transform(self, feats, idx_layer):
        num_layers = len(self.layers)

        if num_layers == 0:
            raise Exception("There are no layers in CascadeForest!")

        if idx_layer >= num_layers:
            raise Exception("Tried to transform features on layer that is not in this CascadeForest!")

        return self.layers[idx_layer].transform(feats)

    def predict_ending_layer(self, feats, predict_probabilities=False):
        # TODO: change this for different combinations of 'ending_layer' ('avg', 'stack')
        preds = self.ending_layer.predict_proba(feats) if predict_probabilities else self.ending_layer.predict(feats)

        return preds

    def _pred_proba(self, split_transformed_feats):
        """ Internal method to predict probabilities for specifically shaped 'split_transformed_feats' list.
        :param split_transformed_feats: list
                List, containing transformed features (numpy.ndarrays) for each grain (in MultiGrainedScanning) or list,
                containing numpy.ndarray with raw features (if no grains are used). Each of these feature arrays needs
                to have same number of columns.
        :return: numpy.ndarray
                Class probabilities for each instance.
        """
        num_layers = len(self.layers)

        curr_val_input = split_transformed_feats[0]
        for idx_layer in range(num_layers - 1):
            print("Layer %d... features shape: %s" % (idx_layer, str(curr_val_input.shape)))
            curr_val_feats = self.layers[idx_layer].transform(curr_val_input)

            curr_val_input = np.hstack((split_transformed_feats[idx_layer % len(split_transformed_feats)], curr_val_feats))

        print("Layer %d... features shape: %s" % (num_layers - 1, str(curr_val_input.shape)))
        # do not concatenate features from multi-grained scanning on last layer
        curr_val_feats = self.layers[num_layers - 1].transform(curr_val_input)

        return self.ending_layer.predict_proba(curr_val_feats)


class CascadeLayer:
    def __init__(self, n_rf=2,
                 n_crf=2,
                 n_rsf=0,
                 n_xonf=0,
                 n_estimators_rf=100,
                 n_estimators_crf=100,
                 n_estimators_rsf=100,
                 n_estimators_xonf=100,
                 k_cv=3,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False,
                 keep_models=True):
        """
        Parameters
        ----------
        :param n_rf: int (default: 2)
                Number of random forests in cascade layer.
        :param n_crf: int (default: 2)
                Number of completely random forests in cascade layer.
        # TODO: add `n_rsf` to doc
        # TODO: add `n_xonf` to doc
        # TODO: add separate `n_estimators_...` for each model in doc
        :param k_cv: int (default: 3)
                Parameter for k-fold cross validation.
        :param classes_: list or numpy.ndarray (default: None)
                How should classes be mapped to indices in probability vectors.
        :param random_state: int (default: None)
                The random state for random number generator.
        :param labels_encoded: bool (default: False)
                Will labels in training set already be encoded as stated in 'classes_'?
        :param keep_models: bool (default: True)
                Whether to keep trained models or not. An example of when you do not need to keep models is
                when determining number of optimal layers in the cascade forest.
        """
        self.n_rf, self.rf_estimators = n_rf, []
        self.n_crf, self.crf_estimators = n_crf, []
        self.n_rsf, self.rsf_estimators = n_rsf, []
        self.n_xonf, self.xonf_estimators = n_xonf, []
        self.n_estimators_rf = n_estimators_rf
        self.n_estimators_crf = n_estimators_crf
        self.n_estimators_rsf = n_estimators_rsf
        self.n_estimators_xonf = n_estimators_xonf

        self.k_cv = k_cv
        self.classes_ = np.array(classes_) if classes_ is not None else None
        if random_state is not None:
            np.random.seed(random_state)
        self.labels_encoded = labels_encoded
        self.keep_models = keep_models

        self.idx_fit_next = 0

        self.kfold_acc = None

    def train_layer(self, feats, labels):
        """
            This method is currently not the main focus because caching is not yet implemented - `fit_transform(...)`
            is therefore better suited, as it does not keep/save models in memory and does fitting and predicting
            "simultaneously".
        """
        # TODO: add `feats_rsf`, `feats_xonf`
        feats_crf, feats_rf = [], []

        layer_acc = 0.0

        # train completely random forests
        for idx_crf in range(self.n_crf):
            crf_model = ExtraTreesClassifier(n_estimators=self.n_estimators_crf,
                                             max_features=1,
                                             n_jobs=-1)
            curr_model, curr_feats, curr_acc = common_utils.get_class_distribution(feats=feats,
                                                                                   labels=labels,
                                                                                   model=crf_model,
                                                                                   num_all_classes=self.classes_.shape[0],
                                                                                   k_cv=self.k_cv)

            layer_acc += curr_acc

            if self.keep_models:
                self.crf_estimators.append(curr_model)
            feats_crf.append(curr_feats)

        # TODO: account for `self.n_crf` being 0
        feats_crf = np.hstack(feats_crf)

        # train random forests
        for idx_rf in range(self.n_rf):
            rf_model = RandomForestClassifier(n_estimators=self.n_estimators_rf,
                                              n_jobs=-1)
            curr_model, curr_feats, curr_acc = common_utils.get_class_distribution(feats=feats,
                                                                                   labels=labels,
                                                                                   model=rf_model,
                                                                                   num_all_classes=self.classes_.shape[0],
                                                                                   k_cv=self.k_cv)

            layer_acc += curr_acc

            if self.keep_models:
                self.rf_estimators.append(curr_model)
            feats_rf.append(curr_feats)

        # TODO: account for `self.n_rf` being 0
        feats_rf = np.hstack(feats_rf)

        # TODO: train random subspace forests
        # ...

        # TODO: train random X-of-N forests
        # ...

        # TODO: divide by (n_rf + n_crf + n_rsf + n_xonf)
        layer_acc /= (self.n_rf + self.n_crf)
        self.kfold_acc = layer_acc
        print("Average LAYER accuracy is %f..." % self.kfold_acc)

        # TODO: stack feats_crf, feats_rf, feats_rsf, feats_xonf (be careful to be consistent with other methods)
        return np.hstack((feats_crf, feats_rf))

    def fit_transform(self, train_feats, train_labels, test_feats):
        train_feats_crf, train_feats_rf = [], []
        test_feats_crf, test_feats_rf = [], []
        train_feats_rsf, test_feats_rsf = [], []
        train_feats_xonf, test_feats_xonf = [], []

        all_train, all_test = None, None
        layer_acc = 0.0
        print("Training cascade layer...")

        for idx_crf in range(self.n_crf):
            print("Training CRF#%d..." % idx_crf)
            curr_model = ExtraTreesClassifier(n_estimators=self.n_estimators_crf,
                                              max_features=1,
                                              n_jobs=-1)
            curr_model, curr_train_feats, curr_acc = common_utils.get_class_distribution(feats=train_feats,
                                                                                         labels=train_labels,
                                                                                         model=curr_model,
                                                                                         num_all_classes=self.classes_.shape[0],
                                                                                         k_cv=self.k_cv)

            curr_test_feats = np.zeros((test_feats.shape[0], self.classes_.shape[0]))
            class_indices = curr_model.classes_
            curr_test_feats[:, class_indices] = curr_model.predict_proba(test_feats)

            layer_acc += curr_acc

            train_feats_crf.append(curr_train_feats)
            test_feats_crf.append(curr_test_feats)

        if self.n_crf > 0:
            train_feats_crf = np.hstack(train_feats_crf)
            test_feats_crf = np.hstack(test_feats_crf)

            all_train = train_feats_crf
            all_test = test_feats_crf

        for idx_rf in range(self.n_rf):
            print("Training RF#%d..." % idx_rf)
            curr_model = RandomForestClassifier(n_estimators=self.n_estimators_rf,
                                                n_jobs=-1)
            curr_model, curr_train_feats, curr_acc = common_utils.get_class_distribution(feats=train_feats,
                                                                                         labels=train_labels,
                                                                                         model=curr_model,
                                                                                         num_all_classes=self.classes_.shape[0],
                                                                                         k_cv=self.k_cv)

            curr_test_feats = np.zeros((test_feats.shape[0], self.classes_.shape[0]))
            class_indices = curr_model.classes_
            curr_test_feats[:, class_indices] = curr_model.predict_proba(test_feats)

            layer_acc += curr_acc
            train_feats_rf.append(curr_train_feats)
            test_feats_rf.append(curr_test_feats)

        if self.n_rf > 0:
            train_feats_rf = np.hstack(train_feats_rf)
            test_feats_rf = np.hstack(test_feats_rf)

            if all_train is None:
                all_train = train_feats_rf
                all_test = test_feats_rf
            else:
                all_train = np.hstack((all_train, train_feats_rf))
                all_test = np.hstack((all_test, test_feats_rf))

        for idx_rsf in range(self.n_rsf):
            print("Training RSF#%d..." % idx_rsf)
            curr_model = RandomSubspaceForest(n_estimators=self.n_estimators_rsf,
                                              n_features=int(train_feats.shape[1] ** 0.5))

            curr_model, curr_train_feats, curr_acc = common_utils.get_class_distribution(feats=train_feats,
                                                                                         labels=train_labels,
                                                                                         model=curr_model,
                                                                                         num_all_classes=self.classes_.shape[0],
                                                                                         k_cv=self.k_cv)

            curr_test_feats = np.zeros((test_feats.shape[0], self.classes_.shape[0]))
            class_indices = curr_model.classes_
            curr_test_feats[:, class_indices] = curr_model.predict_proba(test_feats)

            layer_acc += curr_acc
            train_feats_rsf.append(curr_train_feats)
            test_feats_rsf.append(curr_test_feats)

        if self.n_rsf > 0:
            train_feats_rsf = np.hstack(train_feats_rsf)
            test_feats_rsf = np.hstack(test_feats_rsf)

            if all_train is None:
                all_train = train_feats_rsf
                all_test = test_feats_rsf
            else:
                all_train = np.hstack((all_train, train_feats_rsf))
                all_test = np.hstack((all_test, test_feats_rsf))

        for idx_xonf in range(self.n_xonf):
            print("Training XoNF#%d..." % idx_xonf)
            # TODO: `sample_size`, `max_features` parameters (maybe)
            curr_model = RandomXOfNForest(n_estimators=self.n_estimators_xonf)
            curr_model, curr_train_feats, curr_acc = common_utils.get_class_distribution(feats=train_feats,
                                                                                         labels=train_labels,
                                                                                         model=curr_model,
                                                                                         num_all_classes=
                                                                                         self.classes_.shape[0],
                                                                                         k_cv=self.k_cv)

            curr_test_feats = np.zeros((test_feats.shape[0], self.classes_.shape[0]))
            class_indices = curr_model.classes_
            curr_test_feats[:, class_indices] = curr_model.predict_proba(test_feats)

            layer_acc += curr_acc
            train_feats_xonf.append(curr_train_feats)
            test_feats_xonf.append(curr_test_feats)

        if self.n_xonf > 0:
            train_feats_xonf = np.hstack(train_feats_xonf)
            test_feats_xonf = np.hstack(test_feats_xonf)

            if all_train is None:
                all_train = train_feats_xonf
                all_test = test_feats_xonf
            else:
                all_train = np.hstack((all_train, train_feats_xonf))
                all_test = np.hstack((all_test, test_feats_xonf))

        if all_train is None:
            raise Exception("No models were specified for this layer!")

        layer_acc /= (self.n_rf + self.n_crf + self.n_rsf + self.n_xonf)
        self.kfold_acc = layer_acc
        print("Average LAYER accuracy is %f..." % self.kfold_acc)
        print("-------------------------------")

        return all_train, all_test

    def transform(self, feats):
        if not self.keep_models:
            raise Exception("Models were not saved during training. Argument 'keep_models' should be set to True "
                            "when creating a CascadeLayer...")

        feats_crf, feats_rf = [], []

        for idx_model in range(self.n_crf):
            curr_proba_preds = np.zeros((feats.shape[0], self.classes_.shape[0]))
            class_indices = self.crf_estimators[idx_model].classes_
            curr_proba_preds[:, class_indices] = self.crf_estimators[idx_model].predict_proba(feats)

            feats_crf.append(curr_proba_preds)

        # TODO: account for `self.n_crf` being 0
        feats_crf = np.hstack(feats_crf)

        for idx_model in range(self.n_rf):
            curr_proba_preds = np.zeros((feats.shape[0], self.classes_.shape[0]))
            class_indices = self.rf_estimators[idx_model].classes_
            curr_proba_preds[:, class_indices] = self.rf_estimators[idx_model].predict_proba(feats)

            feats_rf.append(curr_proba_preds)

        # TODO: account for `self.n_rf` being 0
        feats_rf = np.hstack(feats_rf)

        # TODO: transform data with random subspace forests
        # ...

        # TODO: account for `self.n_rsf` being 0
        # ...

        # TODO: transform data with random X-of-N forests
        # ...

        # TODO: account for `self.n_xonf` being 0
        # ...

        # TODO: hstack `feats_crf`, `feats_rf`, `feats_rsf`, `feats_xonf`
        return np.hstack((feats_crf, feats_rf))


class EndingLayerAverage:
    def __init__(self, classes_):
        self.classes_ = classes_

    def predict_proba(self, feats):
        num_examples = feats.shape[0]

        # reshape features so that predicted probabilities for same class are in same column
        # e.g. [p11, p12, p13, p21, p22, p23] -> [[p11, p12, p13], [p21, p22, p23]]
        reshaped_feats = np.reshape(feats, [-1, self.classes_.shape[0]])
        split_feats = np.split(reshaped_feats, num_examples)

        return np.mean(split_feats, axis=1)

    def predict(self, feats):
        proba_preds = self.predict_proba(feats)

        return self.classes_[np.argmax(proba_preds, axis=1)]

    def fit(self, feats, labels):
        pass

    def fit_predict(self, train_feats, train_labels, test_feats):
        return self.predict(test_feats)


class EndingLayerStacking:
    def __init__(self, classes_, model=None, k_cv=3):
        self.classes_ = classes_

        self._stacking_model = model if model is not None else LogisticRegression()
        self.k_cv = k_cv
        self._is_fitted = False

    def predict_proba(self, feats):
        if not self._is_fitted:
            raise Exception("Stacking model is not fitted...")

        proba_preds = np.zeros((feats.shape[0], self.classes_.shape[0]))
        class_indices = self._stacking_model.classes_
        proba_preds[:, class_indices] = self._stacking_model.predict_proba(feats)

        return proba_preds

    def predict(self, feats):
        return np.argmax(self.predict_proba(feats), axis=1)

    def fit(self, feats, labels):
        self._is_fitted = False
        self._stacking_model, _, curr_acc = common_utils.get_class_distribution(feats=feats,
                                                                                labels=labels,
                                                                                model=self._stacking_model,
                                                                                num_all_classes=self.classes_.shape[0],
                                                                                k_cv=self.k_cv)

        print("Final layer average accuracy: %.5f..." % curr_acc)
        self._is_fitted = True

    def fit_predict(self, train_feats, train_labels, test_feats):
        # designed not to save the trained model

        model, _, curr_acc = common_utils.get_class_distribution(feats=train_feats,
                                                                 labels=train_labels,
                                                                 model=self._stacking_model,
                                                                 num_all_classes=self.classes_.shape[0],
                                                                 k_cv=self.k_cv)

        proba_preds = np.zeros((test_feats.shape[0], self.classes_.shape[0]))
        class_indices = model.classes_
        proba_preds[:, class_indices] = model.predict_proba(test_feats)

        return self.classes_[np.argmax(proba_preds, axis=1)]
