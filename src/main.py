from csv import reader
import numpy as np
import pandas as pd
import decision_tree

TRAINING_SIZE = 0.8

if __name__ == "__main__":
    np.random.seed(1337)

    with open("../data/iris_data.csv") as iris_dataset:
        df = pd.read_csv(iris_dataset, delimiter=",", header=None)
        df = df.sample(frac=1).reset_index(drop=True)

        training_set_X = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 0: 4])
        training_set_y = np.array(df.iloc[0: int(TRAINING_SIZE * len(df)), 4])

        test_set_X = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 0: 4])
        test_set_y = np.array(df.iloc[int(TRAINING_SIZE * len(df)):, 4])

        dt = decision_tree.DecisionTree(max_depth=3)
        dt.fit(training_set_X, training_set_y)
        dt.traverse()
