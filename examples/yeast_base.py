if __name__ == "__main__":
    import numpy as np

    from gcforest.gc_forest import GrainedCascadeForest
    from gcforest import datasets

    # -----------------------------------------------------
    # Example: base gcForest that only uses cascade forest
    # -----------------------------------------------------
    # This example fits a gcForest model on the YEAST data set. The cascade forest consists of 4 random and 4 completely
    # random forests in each layer.

    train_X, train_y, test_X, test_y = datasets.prep_yeast()
    gcf = GrainedCascadeForest(n_rf_cascade=4,
                               n_crf_cascade=4,
                               n_estimators_rf=500,
                               n_estimators_crf=500,
                               k_cv=3)

    preds = gcf.fit_predict(train_X, train_y, test_X)
    accuracy = np.sum(preds == test_y) / test_y.shape[0]
    print("[Accuracy: %.5f]" % accuracy)
