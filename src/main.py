from csv import reader
import numpy as np
import decision_tree

TRAINING_SIZE = 0.8

if __name__ == "__main__":
    np.random.seed(1337)

    with open("../data/iris_data.csv") as iris_dataset:
        reader_obj = reader(iris_dataset, delimiter=",")

        dataset = np.array(list(reader_obj))
        np.random.shuffle(dataset)

        # Note to self: currently all strings -> split input features and labels
        training_set, test_set = dataset[:int(TRAINING_SIZE * np.size(dataset, axis=0)), :], \
                                 dataset[int(TRAINING_SIZE * np.size(dataset, axis=0)):, :]