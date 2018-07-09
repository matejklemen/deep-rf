import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.externals import joblib
import os
import shutil


def get_class_distribution(feats, labels, model, num_all_classes, k_cv=3):
    """ Gets predicted probabilities for 'feats' using k-fold cross validation and trains a model on entire data set
    afterwards.

    Parameters
    ----------
    :param feats: numpy.ndarray
            Training data - features.
    :param labels: numpy.ndarray
            Training data - labels.
    :param model:
            Model to be trained.
    :param num_all_classes: int
            Number of all classes in entire data set. Required because the pair (feats, labels) might be a data subset
            which does not include all unique labels.
    :param k_cv: int (default: 3)
            Parameter for k-fold cross validation.
    :return: tuple
            (trained model, class distribution, average accuracy) where class distribution has same number of rows as
            'feats' and (num_all_classes) columns.
    """

    kf = StratifiedKFold(n_splits=k_cv, shuffle=True)

    class_distrib = np.zeros((feats.shape[0], num_all_classes))

    avg_acca = 0.0

    # k-fold cross validation to obtain class distribution
    for train_indices, test_indices in kf.split(range(labels.shape[0]), labels):
        model.fit(feats[train_indices, :], labels[train_indices])

        # can't get this to work in 1 step, so here's an ugly workaround
        curr_part = class_distrib[test_indices, :]
        curr_part[:, model.classes_] = model.predict_proba(feats[test_indices])
        class_distrib[test_indices, :] = curr_part

        avg_acca += np.sum(model.classes_[np.argmax(curr_part, axis=1)] == labels[test_indices]) / test_indices.shape[0]

    avg_acca /= k_cv
    print("Average k-fold cross-validation accuracy of a SINGLE ENSEMBLE is %f..." % avg_acca)

    # retrain model on whole training set
    model.fit(feats, labels)

    return model, class_distrib, avg_acca


def create_cache_dir(dir_path):
    # creates directory if it doesn't yet exist - if it does exist, it preserves original directory
    os.makedirs(dir_path, exist_ok=True)


def remove_cache_dir(dir_path):
    # cleans up cache directory
    shutil.rmtree(dir_path, ignore_errors=True)


def save_data(data_obj, path):
    print("Saving data!")
    joblib.dump(value=data_obj,
                filename=path,
                protocol=-1)


def load_data(path):
    print("Loading data!")
    return joblib.load(filename=path)
