import numpy as np

"""
    Splits dataset into a greater-than and lower-than-or-equal part according to
    attribute at index 'split_attr_idx' and threshold value 'split_val'.
    :param dataset (numpy.ndarray)
    :param split_attr_idx (int) index of splitting attribute, needs to be in range [0, ncols of 'dataset')
    :param split_val (float) threshold value
"""
def _split_dataset(dataset, split_attr_idx, split_val):
    return dataset[dataset[:, split_attr_idx] > split_val, :], \
           dataset[dataset[:, split_attr_idx] <= split_val, :]


def _gini_index(labels, poss_classes):
    # 1 - sum_{poss_classes} (p_class ^ 2)
    gidx = 1.0

    for class_ in poss_classes:
        gidx -= (np.sum(labels == class_) / np.size(labels, axis=0)) ** 2

    return gidx

def _gini_impurity(dataset):
    pass


class DecisionTree:
    def __init__(self):
        pass

    def fit(self, data_train, labels_train):
        pass

    def predict(self, data_test):
        pass


class InternalTreeNode:
    def __init__(self):
        self.split_attr_idx, self.split_val = None, None


# contains class prediction
class LeafTreeNode:
    def __init__(self):
        # {class: probability}
        self.probabilities = {}
        self.outcome = None