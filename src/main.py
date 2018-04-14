from csv import reader
from random import seed, shuffle

TRAINING_SIZE = 0.8

if __name__ == "__main__":
    seed(1337)

    with open("../data/iris_data.csv") as iris_dataset:
        reader_obj = reader(iris_dataset, delimiter = ",")

        dataset = [line for line in reader_obj]

        for row_idx in range(len(dataset)):
            for col_idx in range(len(dataset[row_idx])):
                # fifth col is class attribute in this dataset
                if col_idx % 5 < 4:
                    dataset[row_idx][col_idx] = float(dataset[row_idx][col_idx])

        shuffle(dataset)
        training_set, test_set = dataset[:int(TRAINING_SIZE * len(dataset))], dataset[int(TRAINING_SIZE * len(dataset)):]