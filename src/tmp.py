import numpy as np
import gc_forest

if __name__ == "__main__":
    X_train = np.random.random(36) * 100

    gcf = gc_forest.GCForest()
    bins = gcf._kfold_cv(num_examples=100, k=7)
    print(bins)

# Code which takes advantage of multiple processors
# put the second thing into fit() method of random forests
#
# def _prepare_fit(self, input_train, labels_train, num_features, sample_size):
#     # choose sample for current tree (random with replacement)
#     curr_sample_indices = np.random.choice(sample_size, sample_size)
#
#     curr_input = input_train[curr_sample_indices, :]
#     curr_labels = labels_train[curr_sample_indices]
#
#     curr_tree = decision_tree.DecisionTree(label_idx_mapping=self.label_idx_mapping,
#                                            random_state=self.random_state)
#     curr_tree.fit(curr_input, curr_labels, num_features)
#
#     return curr_tree
#
# pool = multiprocessing.Pool()
# tmp_worker = [pool.apply_async(self._prepare_fit, (input_train, labels_train, num_features, sample_size))
#               for _ in range(self.num_trees)]
# self.trees = [res.get() for res in tmp_worker]