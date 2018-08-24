import numpy as np
import multiprocessing
from sklearn.tree import DecisionTreeClassifier
from itertools import chain

# data for parallel fitting of random subspace forests
# READ-ONLY! (write only in main process)
_shared_data = {}


def _set_shared_data(feats, labels, shape):
    _shared_data["feats"] = feats
    _shared_data["labels"] = labels
    _shared_data["shape"] = shape


def _clear_shared_data():
    _shared_data.clear()


class RandomSubspaceForest:
    def __init__(self, n_estimators=100,
                 n_features="sqrt",
                 max_depth=None,
                 min_samples_leaf=1,
                 n_jobs=1,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False):
        self.n_estimators = n_estimators
        self.n_features = n_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = max(1, n_jobs) if n_jobs != -1 else multiprocessing.cpu_count()
        self.classes_ = classes_
        self.random_state = random_state
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

    def _fit_process(self, n_trees, rand_seed):
        """ Internal method that is called in subprocesses to fit a part of all random subspaces in a forest.

        Parameters
        ----------
        n_trees: int
            Number of trees that are to be fitted in this function
        rand_seed: int
            Random seed for repeatability

        Returns
        -------
        (list, list)
            Trained tree objects and corresponding chosen features
        """
        np.random.seed(rand_seed)

        feats = _shared_data["feats"]
        labels = _shared_data["labels"]
        shape = _shared_data["shape"]

        feats = np.frombuffer(feats, np.float32).reshape(shape)
        labels = np.frombuffer(labels, np.int32)

        num_all_feats = shape[1]
        trees, chosen_feats = [], []
        for i in range(n_trees):
            selected_features = np.random.choice(num_all_feats, self._n_features, replace=True).tolist()
            chosen_feats.append(selected_features)

            dt = DecisionTreeClassifier(max_depth=self.max_depth,
                                        min_samples_leaf=self.min_samples_leaf)
            dt.fit(feats[:, selected_features], labels)
            trees.append(dt)

        return trees, chosen_feats

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

        # put features and labels into shared data
        feats_shape = feats.shape
        feats_base = multiprocessing.Array("f", feats_shape[0] * feats_shape[1], lock=False)
        feats_np = np.frombuffer(feats_base, dtype=np.float32).reshape(feats_shape)
        np.copyto(feats_np, feats)

        labels_base = multiprocessing.Array("I", feats_shape[0], lock=False)
        labels_np = np.frombuffer(labels_base, dtype=np.int32)
        np.copyto(labels_np, labels)

        with multiprocessing.Pool(processes=self.n_jobs,
                                  initializer=_set_shared_data,
                                  initargs=(feats_base, labels_base, feats_shape)) as pool:
            async_objs = []
            for idx_proc in range(self.n_jobs):
                # divide `n_estimators` between `self.n_jobs` processes -
                # the int() rounding of floats makes sure that work gets split as evenly as possible
                start = int(float(idx_proc) * self.n_estimators / self.n_jobs)
                end = int(float(idx_proc + 1) * self.n_estimators / self.n_jobs)

                async_objs.append(pool.apply_async(func=self._fit_process,
                                                   args=(end - start, np.random.randint(2**30))))

            res = [obj.get() for obj in async_objs]
            self.estimators = list(chain(*[est for est, _ in res]))
            self._chosen_features = np.array(list(chain(*[chosen for _, chosen in res])))

        _clear_shared_data()
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
