import numpy as np
import common_utils
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


class MultiGrainedScanning:
    def __init__(self, grains=None):
        """
        Parameters
        ----------
        :param grains: list (default: None)
                Grain objects in sequential order
        """
        self.grains = grains if grains is not None else []
        self.idx_fit_next = 0

    def _sanity_check_grains(self, sudo):
        if len(self.grains) == 0:
            raise Exception("There are no grains in the multi-grained structure!")

        if not sudo and self.idx_fit_next == len(self.grains):
            raise Exception("All grains are already trained. If you wish to ignore this warning,"
                            "set parameter sudo=True")

    def add_grain(self, grain):
        """ Insert grain into the multi grain structure.
            NOTE: this method only inserts the grain, it does not train the grain classifiers or anything else.

        Parameters
        ----------
        :param grain: Grain
                Grain object
        :return: None
        """
        if not isinstance(grain, Grain):
            raise Exception("'grain' must be an object of type Grain!")

        self.grains.append(grain)

    def train_next_grain(self, feats, labels):
        """ Trains next layer in multi grained scanning structure and returns obtained features.

        Parameters
        ----------
        :param feats: numpy.ndarray
                Features (i.e. X) for the models of a Grain to be trained on.
        :param labels: numpy.ndarray
                Labels (i.e. y) for the models of a Grain to be trained on.
        :return: numpy.ndarray
                Predictions obtained from cross-validation.
        """
        self._sanity_check_grains(sudo=False)

        transformed_feats, labels = self.grains[self.idx_fit_next].create(feats, labels)
        self.idx_fit_next += 1

        return transformed_feats

    def train_all_grains(self, feats, labels, sudo=False):
        """ Trains all layers in multi grained scanning structure and returns all obtained features.

        Parameters
        ----------
        :param feats: numpy.ndarray
                Features (i.e. X) for the models of a Grain to be trained on.
        :param labels: numpy.ndarray
                Labels (i.e. y) for the models of a Grain to be trained on.
        :param sudo: bool (default: False)
                Flag that specifies whether to retrain (override) layers if they are already trained.
        :return: list
                List containing feature and labels numpy.ndarrays for each grain (in same order as specified when
                constructing MultiGrainedScanning object or when inserting grains with add_grain(...)).
        """
        self._sanity_check_grains(sudo=sudo)

        self.idx_fit_next = 0
        transformed_feats = []

        for idx_grain in range(len(self.grains)):
            transformed_feats.append(self.grains[idx_grain].create(feats, labels))
            self.idx_fit_next += 1

        return transformed_feats

    def transform(self, idx_layer, feats):
        """ Transforms features with grain on layer 'idx_layer'.

        Parameters
        ----------
        :param idx_layer: int
                Index of layer in multi-grained structure.
        :param feats: numpy.ndarray
                Features (i.e. X) to be transformed.
        :return: numpy.ndarray
                Transformed features.
        """
        if len(self.grains) == 0:
            raise Exception("There are no grains in the multi-grained structure!")

        if idx_layer > len(self.grains):
            raise Exception("Grain %d does not exist (is out of bounds)!" % idx_layer)

        if idx_layer >= self.idx_fit_next:
            raise Exception("Grain %d has not been trained yet!" % idx_layer)

        return self.grains[idx_layer].transform(feats)

    def transform_all_grains(self, feats):
        """ Transform features with all grains in multi-grained structure.

        Parameters
        ----------
        :param feats: numpy.ndarray
                Features (i.e. X) to be transformed.
        :return: list
                List with >=1 numpy.ndarrays containing transformed features
        """

        transformed_feats = []

        for idx_grain in range(len(self.grains)):
            transformed_feats.append(self.transform(idx_grain, feats))

        return transformed_feats


class Grain:
    def __init__(self, window_size,
                 single_shape,
                 stride=1,
                 n_rf=1,
                 n_crf=1,
                 n_estimators=100,
                 k_cv=3,
                 classes_=None,
                 random_state=None,
                 labels_encoded=False):
        """
        Parameters
        ----------
        :param window_size: int or tuple or list or numpy.ndarray
                Window size, used in sliding window - format for 2D windows: [num_rows, num_cols].
        :param single_shape: int or tuple or list or numpy.ndarray
                Shape of a single example, before being unrolled (i.e. flattened into a 1D vector) - format:
                [num_rows, num_cols] i.e. [height, width].
        :param stride: int or tuple or list or numpy.ndarray (default: 1)
                Step size for sliding window (int/tuple/list/np.ndarray) - format for 2D stride shape:
                [num_rows, num_cols].
        :param n_rf: int (default: 1)
                Number of random forests trained on sliced data.
        :param n_crf: int (default: 1)
                Number of completely random forests trained on sliced data.
        :param n_estimators: int (default: 100)
                Number of trees in a single forest.
        :param k_cv: int (default: 3)
                Parameter for k-fold cross validation.
        :param classes_: list or numpy.ndarray (default: None)
                How should classes be mapped to indices in probability vectors.
        :param random_state: int (default: None)
                The random state for random number generator.
        :param labels_encoded: bool (default: False)
                Will labels in training set already be encoded as stated in 'classes_'?
        """
        self.wind_size = self._process(window_size)
        self.stride = self._process(stride)
        self.single_shape = self._process(single_shape)
        self.n_rf, self.rf_estimators = n_rf, []
        self.n_crf, self.crf_estimators = n_crf, []
        self.n_estimators = n_estimators
        self.k_cv = k_cv
        self.classes_ = np.array(classes_) if classes_ is not None else None
        if random_state is not None:
            np.random.seed(random_state)
        self.labels_encoded = labels_encoded

        self.kfold_acc = None

    @staticmethod
    def _process(shape_el):
        # convert scalar window sizes to np.ndarrays to be able to make slicing method more general
        if isinstance(shape_el, int):
            wind_size = np.array([1, shape_el])
        elif isinstance(shape_el, tuple):
            wind_size = np.array([*shape_el])
        elif isinstance(shape_el, list):
            wind_size = np.array(shape_el)
        elif not isinstance(shape_el, np.ndarray):
            raise Exception("Shape needs to be int/tuple/list/np.ndarray...")

        return wind_size

    def slice_data(self, features):
        """ Applies sliding window, specified when constructing this grain.
        WARNING: This can be very memory intensive for a larger data set. You might want to consider setting stride to
        something else than 1 to avoid running out of memory.

        Parameters
        ----------
        :param features: numpy.ndarray
                Features, on which sliding window will be applied.
        :return: numpy.ndarray
                Sliced features.
        """
        # convert 1d array to 2d to avoid some branching and improve readability by just a bit
        if len(features.shape) == 1:
            features = np.expand_dims(features, 0)

        wind_single_row = np.arange(self.wind_size[1])
        wind_all_rows = np.tile(wind_single_row, (self.wind_size[0], 1))

        # take same columns in all rows of sliding window - makes use of broadcasting
        wind_all_rows += np.reshape(np.arange(self.wind_size[0]), [-1, 1]) * self.single_shape[1]
        wind_all_rows = wind_all_rows.flatten()

        iters_cols = np.reshape(np.arange(start=0, stop=(self.single_shape[1] - self.wind_size[1] + 1), step=self.stride[1]), [-1, 1])
        # create indices for when sliding window gets moved right by step self.stride[1] (for each of these movements)
        all_winds_single_example = wind_all_rows + iters_cols
        all_winds_single_example = all_winds_single_example.flatten()

        iters_rows = np.reshape(np.arange(start=0, stop=(self.single_shape[0] - self.wind_size[0] + 1), step=self.stride[0]), [-1, 1])
        # create indices for when sliding window gets moved down by step self.stride[0] (for each of these movements)
        all_winds_single_example = all_winds_single_example + iters_rows * self.single_shape[1]

        return features[:, all_winds_single_example].flatten().reshape([-1, self.wind_size[0] * self.wind_size[1]])

    def create(self, features, labels):
        sliced_data = self.slice_data(features)
        print("Successfully sliced data for window size %s and stride %s ----> shape of slices: %s..." %
              (str(self.wind_size), str(self.stride), str(sliced_data.shape)))

        # because labels do not get appended to sliced features in slice_data, it is done here
        multiply_factor = int(sliced_data.shape[0] / features.shape[0])
        labels = np.tile(np.reshape(labels, [-1, 1]), (1, multiply_factor)).flatten()

        feats_crf, feats_rf = [], []

        layer_acc = 0.0

        for idx_crf in range(self.n_crf):
            crf_model = ExtraTreesClassifier(n_estimators=self.n_estimators,
                                             max_features=1,
                                             n_jobs=-1)

            print("Training completely random forest %d with %d estimators..." % (idx_crf + 1, self.n_estimators))
            trained_crf, curr_proba_preds, curr_acc = common_utils.get_class_distribution(feats=sliced_data,
                                                                                labels=labels,
                                                                                model=crf_model,
                                                                                num_all_classes=self.classes_.shape[0],
                                                                                k_cv=self.k_cv)

            layer_acc += curr_acc

            # combine predictions for slices of same example together
            feats_crf.append(curr_proba_preds.reshape([-1, multiply_factor * self.classes_.shape[0]]))
            # save trained model
            self.crf_estimators.append(trained_crf)

        feats_crf = np.hstack(feats_crf)

        for idx_rf in range(self.n_rf):
            rf_model = RandomForestClassifier(n_estimators=self.n_estimators,
                                              n_jobs=-1)

            print("Training random forest %d with %d estimators..." % (idx_rf + 1, self.n_estimators))
            trained_rf, curr_proba_preds, curr_acc = common_utils.get_class_distribution(feats=sliced_data,
                                                                               labels=labels,
                                                                               model=rf_model,
                                                                               num_all_classes=self.classes_.shape[0],
                                                                               k_cv=self.k_cv)

            layer_acc += curr_acc

            # combine predictions for slices of same example together
            feats_rf.append(curr_proba_preds.reshape([-1, multiply_factor * self.classes_.shape[0]]))
            # save trained model
            self.rf_estimators.append(trained_rf)

        feats_rf = np.hstack(feats_rf)

        layer_acc /= (self.n_rf + self.n_crf)
        self.kfold_acc = layer_acc
        print("Average LAYER accuracy is %f..." % self.kfold_acc)

        return np.hstack((feats_crf, feats_rf))

    def transform(self, features):
        sliced_data = self.slice_data(features)

        multiply_factor = int(sliced_data.shape[0] / features.shape[0])
        feats_crf, feats_rf = [], []

        for idx_crf in range(self.n_crf):
            curr_proba_preds = np.zeros((sliced_data.shape[0], self.classes_.shape[0]))
            class_indices = self.crf_estimators[idx_crf].classes_
            curr_proba_preds[:, class_indices] = self.crf_estimators[idx_crf].predict_proba(sliced_data)

            # combine predictions for slices of same example together
            feats_crf.append(curr_proba_preds.reshape([-1, multiply_factor * self.classes_.shape[0]]))

        feats_crf = np.hstack(feats_crf)

        for idx_rf in range(self.n_rf):
            curr_proba_preds = np.zeros((sliced_data.shape[0], self.classes_.shape[0]))
            class_indices = self.rf_estimators[idx_rf].classes_
            curr_proba_preds[:, class_indices] = self.rf_estimators[idx_rf].predict_proba(sliced_data)

            # combine predictions for slices of same example together
            feats_rf.append(curr_proba_preds.reshape([-1, multiply_factor * self.classes_.shape[0]]))

        feats_rf = np.hstack(feats_rf)

        return np.hstack((feats_crf, feats_rf))
