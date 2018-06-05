import numpy as np
import pandas as pd
import gc_forest
import time

TRAINING_SIZE = 0.8
VALIDATION_SIZE = 0.2
RAND_STATE = 1


def prep_iris():
    with open("../data/iris_data.csv") as iris_dataset:
        df = pd.read_csv(iris_dataset, delimiter=",", header=None)
        df = df.sample(frac=1).reset_index(drop=True)

        training_set_X = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 0: 4])
        training_set_y = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 4])

        test_set_X = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 0: 4])
        test_set_y = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 4])

        return training_set_X, training_set_y, test_set_X, test_set_y


def prep_yeast():
    with open("../data/yeast.data") as yeast_dataset:
        df = pd.read_table(yeast_dataset, delimiter=" ", header=None)
        df = df.sample(frac=1).reset_index(drop=True)

        training_set_X = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 0: 8])
        training_set_y = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 8])

        test_set_X = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 0: 8])
        test_set_y = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 8])

        return training_set_X, training_set_y, test_set_X, test_set_y

def prep_adult():
    with open("../data/adult.data") as yeast_train, open("../data/adult.test") as yeast_test:
        training_set = pd.read_csv(yeast_train, header=None, na_values=" ?")
        test_set = pd.read_csv(yeast_test, header=None, na_values=" ?")

        training_set_X, training_set_y = training_set.iloc[:, 0: 14], training_set.iloc[:, 14]
        test_set_X, test_set_y = test_set.iloc[:, 0: 14], test_set.iloc[:, 14]

        return training_set_X, training_set_y, test_set_X, test_set_y


# TODO
def prep_imdb():
    pass


if __name__ == "__main__":
    np.random.seed(RAND_STATE)
    training_set_X, training_set_y, test_set_X, test_set_y = prep_yeast()

    # print(np.unique(np.array(["classA", "classB", "classC"])))
    # print(np.unique(np.array(["classA", "classB", "classC"]), return_inverse=True))

    tic = time.perf_counter()

    gcf = gc_forest.GCForest(window_sizes=[3, 5, 7], k_cv=5, n_estimators=100)
    gcf.fit(training_set_X, training_set_y)

    preds_test = gcf.predict(test_set_X)

    toc = time.perf_counter()
    print("Elapsed time: %fs" % (toc - tic))

    acc_test = np.sum(preds_test == test_set_y) / test_set_y.shape[0]
    print("Accuracy [test]: %f" % acc_test)