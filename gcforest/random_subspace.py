import numpy as np
from sklearn.tree import DecisionTreeClassifier


class RandomSubspaceForest:
    def __init__(self, n_estimators=100,
                 n_features=None,
                 max_depth=None,
                 min_samples_leaf=1,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.classes_ = classes_
        self.labels_encoded = labels_encoded

        if random_state is not None:
            np.random.seed(random_state)

        self.estimators = []
        self._n_features = None
        # numbers in i-th row corresponds to features selected for training i-th tree 
        self._chosen_features = []
        self._is_fitted = False

    @staticmethod
    def calc_n_feats(state, n_all_feats):
        if state is None:
            return n_all_feats
        elif isinstance(state, int):
            return state
        elif isinstance(state, float):
            return int(state * n_all_feats)
        elif state in ("auto", "sqrt"):
            return int(np.sqrt(n_all_feats))
        elif state == "log2":
            return int(np.log2(n_all_feats))
        else:
            raise ValueError("Invalid 'max_features' value encountered (%s)..." % str(state))

    def _assign_labels(self, labels_train):
        if self.classes_ is None:
            # wanted classes not provided
            self.classes_, encoded_labels = np.unique(labels_train, return_inverse=True)
        else:
            encoded_labels = np.zeros_like(labels_train, np.int32)
            for encoded_label in range(self.classes_.shape[0]):
                encoded_labels[labels_train == self.classes_[encoded_label]] = encoded_label

        return encoded_labels

    def fit(self, feats, labels):
        if feats.ndim == 1:
            feats = np.expand_dims(feats, 0)

        # assign mapping from class label to index in probability vector
        if not self.labels_encoded:
            labels = self._assign_labels(labels)

        # clear existing data if it exists
        self._is_fitted = False
        self.estimators = []
        self._chosen_features = []
        num_all_feats = feats.shape[1]

        self._n_features = RandomSubspaceForest.calc_n_feats(self.n_features, num_all_feats)

        for idx_tree in range(self.n_estimators):            
            selected_features = np.random.choice(num_all_feats, self._n_features, replace=True)
            self._chosen_features.append(selected_features)

            dt = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf)
            dt.fit(feats[:, selected_features], labels)
            self.estimators.append(dt)

        self._chosen_features = np.array(self._chosen_features)
        self._is_fitted = True

    def predict_proba(self, feats):
        if not self._is_fitted:
            raise Exception("RandomSubspaceForest is not fitted!")

        proba_preds = np.zeros((feats.shape[0], self.classes_.shape[0]), dtype=np.float32)

        for idx_tree in range(self.n_estimators):
            class_indices = self.estimators[idx_tree].classes_
            selected_features = self._chosen_features[idx_tree, :]

            proba_preds[:, class_indices] += self.estimators[idx_tree].predict_proba(feats[:, selected_features])

        proba_preds /= self.n_estimators

        return proba_preds

    def predict(self, feats):
        return self.classes_[np.argmax(self.predict_proba(feats=feats), axis=1)]
