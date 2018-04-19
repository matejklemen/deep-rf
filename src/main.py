import numpy as np
import pandas as pd
import decision_tree
import random_forest

TRAINING_SIZE = 0.8
RAND_STATE = 1

if __name__ == "__main__":
    np.random.seed(RAND_STATE)
    with open("../data/iris_data.csv") as iris_dataset:
        df = pd.read_csv(iris_dataset, delimiter=",", header=None)
        df = df.sample(frac=1).reset_index(drop=True)

        training_set_X = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 0: 4])
        training_set_y = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 4])

        test_set_X = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 0: 4])
        test_set_y = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 4])

        rf = random_forest.RandomForest(num_trees=100, random_state=RAND_STATE)
        rf.fit(training_set_X, training_set_y)

        preds_test = rf.predict(test_set_X)
        acc_test = (np.sum(preds_test == test_set_y) / np.size(test_set_y))

        print("Accuracy [test]: %f" % acc_test)