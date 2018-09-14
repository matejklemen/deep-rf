if __name__ == "__main__":
    import numpy as np

    from gcforest.gc_forest import GrainedCascadeForest
    from gcforest import datasets

    # ---------------------------------------------------------------------------
    # Example: base gcForest that uses multi-grained scanning and cascade forest
    # ---------------------------------------------------------------------------
    # This example fits a gcForest model on a subset of MNIST data set (15K examples, 10K of those used for training
    # and 5K for testing). It uses 1 random forest and 1 completely random forest for each of the 3 grain sizes used
    # (7x7, 10x10 and 13x13) in multi-grained scanning part of gcForest. 2x2 stride is used for each of the grain sizes
    # to reduce the size of features, created in multi-grained scanning (e.g. "subsampling"). The cascade forest
    # consists of 4 random and 4 completely random forests in each layer.

    train_X, train_y, test_X, test_y = datasets.prep_mnist_org_paper(mode="medium")
    gcf = GrainedCascadeForest(single_shape=[28, 28],  # needs to be specified because image vectors are unrolled
                               n_rf_grain=1,
                               n_crf_grain=1,
                               n_rf_cascade=4,
                               n_crf_cascade=4,
                               window_sizes=[(7, 7), (10, 10), (13, 13)],
                               strides=[(2, 2), (2, 2), (2, 2)],
                               n_estimators_rf=500,
                               n_estimators_crf=500,
                               k_cv=3)

    preds = gcf.fit_predict(train_X, train_y, test_X)
    accuracy = np.sum(preds == test_y) / test_y.shape[0]
    print("[Accuracy: %.5f]" % accuracy)
