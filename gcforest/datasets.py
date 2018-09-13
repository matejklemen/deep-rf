import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_olivetti_faces
import os

# absolute path to data folder
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "")


def prep_yeast():
    with open(DATA_DIR + "yeast.data") as yeast_dataset:
        data = np.genfromtxt(yeast_dataset, delimiter=" ", dtype=None, encoding="utf8")
        data = np.array([[*row] for row in data])

        data_X = data[:, 0: 8].astype(dtype=np.float32)
        data_y = data[:, 8]

        # train_size=0.7 as used in original paper
        train_idx, test_idx = train_test_split(np.arange(data.shape[0]), train_size=0.7, stratify=data_y, random_state=0)

        training_set_X, training_set_y = data_X[train_idx, :], data_y[train_idx]
        test_set_X, test_set_y = data_X[test_idx, :], data_y[test_idx]

        return training_set_X, training_set_y, test_set_X, test_set_y


def prep_adult():
    with open(DATA_DIR + "adult.data") as adult_train, open(DATA_DIR + "adult.test") as adult_test:
        training_set = np.genfromtxt(adult_train, delimiter=", ", dtype=None, encoding="utf8")
        training_set = np.array([[*row] for row in training_set])

        test_set = np.genfromtxt(adult_test, delimiter=", ", dtype=None, encoding="utf8")
        test_set = np.array([[*row] for row in test_set])

        is_categorical = np.array([False,  # Age
                                   True,  # Workclass
                                   False,  # Fnlwgt
                                   True,  # Education
                                   False,  # Education-num
                                   True,  # Marital-status
                                   True,  # Occupation
                                   True,  # Relationship
                                   True,  # Race
                                   True,  # Sex
                                   False,  # Capital-gain
                                   False,  # Capital-loss
                                   False,  # Hours-per-week
                                   True])  # Native-country

        training_set_y = training_set[:, 14]
        test_set_y = test_set[:, 14]

        # go through all input attributes of data set - convert continuous attributes to floats and encode categorical
        # attributes as one-hot attributes
        for i in range(training_set.shape[1] - 1):
            if is_categorical[i]:
                uniqs = np.unique(np.append(training_set[:, i], "?"))
                training_set = np.hstack(
                    (training_set, (uniqs == training_set[:, i].reshape([-1, 1])).astype(np.int32)))
                test_set = np.hstack((test_set, (uniqs == test_set[:, i].reshape([-1, 1])).astype(np.int32)))

        # delete non-encoded categorical values
        training_set_X = np.delete(training_set, [1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1)
        test_set_X = np.delete(test_set, [1, 3, 5, 6, 7, 8, 9, 13, 14], axis=1)

        return training_set_X.astype(dtype=np.float32), training_set_y, test_set_X.astype(dtype=np.float32), test_set_y


def prep_mnist_org_paper(mode="small"):
    # using keras just for datasets is pretty overkill, but it's better than saving all the data sets with the project
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    if mode == "small":
        X_train, y_train = X_train[:2000, :, :], y_train[:2000]
        X_test, y_test = X_test[:1000, :, :], y_test[:1000]
    elif mode == "medium":
        X_train, y_train = X_train[:10000], y_train[:10000]
        X_test, y_test = X_test[:5000], y_test[:5000]

    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1] * X_train.shape[2]])
    X_test = np.reshape(X_test, [X_test.shape[0], X_test.shape[1] * X_test.shape[2]])

    X_train = np.divide(X_train, 255)
    X_test = np.divide(X_test, 255)

    return X_train, y_train, X_test, y_test


def prep_letter():
    with open(DATA_DIR + "letter-recognition.data") as letter_dataset:
        data = np.genfromtxt(letter_dataset, delimiter=",", dtype=None, encoding="utf8")
        data = np.array([[*row] for row in data])

        data_X = data[:, 1:].astype(np.float32)
        data_y = data[:, 0]

        training_set_X, training_set_y = data_X[:16000, :], data_y[:16000]
        test_set_X, test_set_y = data_X[16000:, :], data_y[16000:]

        return training_set_X, training_set_y, test_set_X, test_set_y


def prep_orl(train_imgs_person):
    # train_imgs_person... how many pictures per person will be used for training
    faces = fetch_olivetti_faces(download_if_missing=True)
    # Note: 1 - train_imgs_person/10 would cause problems with `train_imgs_person = 7` (floating point error)
    test_imgs_prop = (10 - train_imgs_person) / 10

    train_X, test_X, train_y, test_y = train_test_split(faces.data, faces.target,
                                                        test_size=test_imgs_prop,
                                                        stratify=faces.target,
                                                        shuffle=True, random_state=0)

    return train_X, train_y, test_X, test_y
