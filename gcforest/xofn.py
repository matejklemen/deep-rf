import numpy as np


class XOfNAttribute(object):
    __slots__ = ('idx_attr', 'thresh_val', 'split_val', 'cost')
    """
    Parameters
    ----------
    :param idx_attr: int or list of ints
            Indices of attributes that make up the X-of-N attribute. If None, initializes an empty attribute.
    :param thresh_val:
            Threshold values, corresponding to attribute indices. If None, initializes an empty attribute.
    """
    def __init__(self, idx_attr=None, thresh_val=None, split_val=None, cost=None):
        self.idx_attr = idx_attr if idx_attr is not None else []
        self.thresh_val = thresh_val if thresh_val is not None else []
        self.split_val = split_val
        self.cost = cost

        if not isinstance(self.idx_attr, list):
            self.idx_attr = [self.idx_attr]

        if not isinstance(self.thresh_val, list):
            self.thresh_val = [self.thresh_val]

    def append_av(self, idx_attr, thresh_val):
        self.idx_attr.append(idx_attr)
        self.thresh_val.append(thresh_val)

    def remove_av(self, idx=None):
        # if idx=None -> removes last
        return self.idx_attr.pop(idx), self.thresh_val.pop(idx)

    def __len__(self):
        return len(self.idx_attr)

    def __str__(self):
        return "XoN(%s, split_val=%d)" \
               % (",".join([str(val) for val in zip(self.idx_attr, self.thresh_val)]), self.split_val)


def _find_valid_values(feat_subset, target):
    if feat_subset.shape[0] < 2:
        return feat_subset

    sort_idx = np.argsort(feat_subset)
    _feats = feat_subset[sort_idx]
    _target = target[sort_idx]
    valid = [_feats[0]]

    for i in range(1, _feats.shape[0]):
        if _target[i] != _target[i - 1] and _feats[i] != valid[-1]:
            valid.append(_feats[i])

    return np.array(valid)


def _fib(n):
    """ Computes Fibonacci's number F(n). If n is a np.array, computes Fibonacci's number for each of the elements.

    Parameters
    ----------
    n: int or np.array

    Returns
    -------
    int or np.array

    Notes
    -----
        Should not be used for numbers above 70-ish, due to this method using Binet's formula which is constrained by
        accuracy of floating point representation.
    """
    phi = (1 + np.sqrt(5)) / 2
    return np.divide(np.power(phi, n) - np.power(- phi, np.negative(n)), np.sqrt(5))


def _apply_attr(train_feats, valid_attrs, valid_thresh):
    """Returns result of applying X-of-N attribute to data set `train_feats`.

    Parameters
    ----------
    train_feats: np.array
        Data set to apply X-of-N attribute to.
    valid_attrs: list or np.array
        Attributes that are used in X-of-N attribute.
    valid_thresh: list or np.array
        Thresholds to go along with `valid_attrs` in X-of-N attribute.

    Returns
    -------
    np.array
        Result of applying X-of-N attribute to data set. Value at index `i` represents how many conditions of X-of-N
        attribute were true in `i`-th row of `train_feats`.
    """
    if train_feats.ndim == 1:
        train_feats = np.expand_dims(train_feats, 0)

    return np.sum(np.less(train_feats[:, valid_attrs], valid_thresh), axis=1)


def _eval_attr(curr_gini, best_gini, best_compl, train_feats, attr_feats, attr_thresh, available_attrs):
    """Evaluates if newly constructed X-of-N attribute either:
    (1) decreases gini index value and has equal/lower complexity than previous best attribute or
    (2) has approx. equal gini index value and has lower complexity than previous best attribute.

    Parameters
    ----------
    curr_gini: float

    best_gini: float

    best_compl: int

    train_feats: np.array

    attr_feats: list or np.array

    attr_thresh: list or np.array

    available_attrs: list or np.array

    Returns
    -------
    float or None
        Complexity of new attribute if new attribute is "better" or None if it is not
    """
    # TODO: decide whether to keep the complexity check or not
    if curr_gini < best_gini:
        curr_compl = _calc_attr_cost(train_feats, attr_feats, attr_thresh, available_attrs=available_attrs)
        # if curr_compl <= best_compl:
        #     return curr_compl
        return curr_compl

    elif np.isclose(curr_gini, best_gini):
        curr_compl = _calc_attr_cost(train_feats, attr_feats, attr_thresh, available_attrs=available_attrs)
        # if curr_compl < best_compl:
        #     return curr_compl
        return curr_compl


def _calc_attr_cost(train_feats, idx_attr, thresh_val, available_attrs):
    """ Calculate new X-of-N attribute's complexity (equation 1 and 2 in paper on X-of-N trees [1]).

    Parameters
    ----------
    train_feats: np.array

    idx_attr: list or np.array

    thresh_val: list or np.array

    available_attrs: list or np.array

    Returns
    -------
    float
        Cost (complexity) of attribute.


    References
    ----------
    [1] Zheng, Z. (2000). Constructing X-of-N attributes for decision tree learning.
        Machine learning, 40(1), 35-75.
    """
    if train_feats.ndim == 1:
        train_feats = np.expand_dims(train_feats, 0)

    unique_attrs_xon = np.unique(idx_attr)

    # N... number of different attributes in X-of-N attribute
    n_unique_attrs_xon = unique_attrs_xon.shape[0]
    # Na... number of primitive attrs. available for creating X-of-N attribute
    n_all_attrs = available_attrs.shape[0]
    # Nvj... number of different values of attribute j
    n_vals = np.array([np.unique(train_feats[:, i]).shape[0] for i in unique_attrs_xon])
    # nj... number of different values of attribute j that appear in X-of-N attribute
    n_unique_vals = np.array([np.unique(np.compress(np.equal(idx_attr, i), thresh_val)).shape[0]
                              for i in unique_attrs_xon])

    # log2(Na) + nj * log2(Nvj) - log2(nj!)
    cost_attr_wise = np.log2(n_all_attrs) + n_unique_vals * np.log2(n_vals) - np.log2(_fib(n_unique_vals))

    return np.sum(cost_attr_wise) - np.log2(_fib(n_unique_attrs_xon))


def _gini(class_dist, num_el):
    """ Computes gini index value.

    Parameters
    ----------
    class_dist: list or np.array
        Number of elements for each class.
    num_el: int
        Number (sum) of all elements in `class_dist`.

    Returns
    -------
    float
        Gini index value.
    """
    return 1 - np.sum(np.square(np.divide(class_dist, num_el)))


def _res_gini_numerical(feat, target, sorted_thresholds=None):
    """ Assumption: works only for numerical feature values. """
    # how examples are distributed among classes prior to checking splits
    uniq_classes, target, class_dist = np.unique(target, return_counts=True, return_inverse=True)

    if uniq_classes.shape[0] == 1:
        # pure subset
        return 0, 0

    if sorted_thresholds is None:
        sorted_thresholds = _find_valid_values(feat, target)
        # np.unique(feat)

    # sort examples (and corresponding labels) by attribute values (i.e. by thresholds)
    sort_indices = np.argsort(feat)
    sorted_feat, sorted_target = feat[sort_indices], target[sort_indices]

    idx_thresh, idx_example = 0, 0
    num_examples = sorted_feat.shape[0]
    best_gini, idx_best_thresh = 1, 0

    # distribution of elements LT/GTE current threshold
    left, left_count = np.zeros_like(class_dist), 0
    right, right_count = np.copy(class_dist), num_examples

    while idx_thresh < sorted_thresholds.shape[0]:
        if sorted_feat[idx_example] < sorted_thresholds[idx_thresh]:
            left[sorted_target[idx_example]] += 1
            right[sorted_target[idx_example]] -= 1

            left_count += 1
            right_count -= 1
            idx_example += 1
        else:
            left_prob = (left_count / num_examples)

            # calculate gini for curr threshold
            curr_gini_res = left_prob * _gini(left, left_count) + (1 - left_prob) * _gini(right, right_count)

            if curr_gini_res < 10e-6:
                # clean subset
                best_gini, idx_best_thresh = curr_gini_res, idx_thresh
                break

            if curr_gini_res < best_gini:
                best_gini, idx_best_thresh = curr_gini_res, idx_thresh

            idx_thresh += 1

    return best_gini, idx_best_thresh


def search_xofn(train_feats, train_labels, available_attrs, last_xon, op_del, available_thresh=None):
    """
    Parameters
    ----------
    train_feats: np.array

    train_labels: np.array

    available_attrs: list or np.array

    last_xon: XofNAttribute
        Last constructed X-of-N attribute prior to this call.
    op_del: bool
        A flag, specifying whether deleting an attribute from 'last_xofn' should be performed. If False,
        insertion of a new (attribute, threshold) will be performed instead.
    available_thresh: list
        Thresholds that will be considered when trying to add (`op_del=False`) an (attr, thresh) pair to X-of-N
        attribute.

    Returns
    -------
    (XOfNAttribute, float) or (None, float)

    """
    # somehow only a single example made it in here
    if train_feats.ndim == 1:
        train_feats = np.expand_dims(train_feats, 0)

    last_xon_vals = _apply_attr(train_feats=train_feats,
                                valid_attrs=np.array(last_xon.idx_attr),
                                valid_thresh=np.array(last_xon.thresh_val))
    splits = np.unique(last_xon_vals)
    prior_gini, ovr_best_thresh = _res_gini_numerical(feat=last_xon_vals,
                                                      target=train_labels,
                                                      sorted_thresholds=splits)
    # print("Prior gini value: %.5f" % prior_gini)
    # gini value and complexity of best newly created X-of-N attribute
    ovr_best_gini, ovr_best_compl = prior_gini, last_xon.cost
    # split point on evaluated X-of-N attributes (i.e. best split for how many conditions are true in X-of-N attr.)
    split_val = 0
    # index of attribute that should be added or deleted (depending on op_del)
    idx_best_attr = 0
    # newly constructed attribute - if it remains None, no better attribute could be constructed
    new_attr = None

    if op_del:
        # try deleting one attribute
        xon_attrs = last_xon.idx_attr
        for idx_attr in range(len(xon_attrs)):
            # take everything but (attr, val) on index `idx_attr`
            mask = np.not_equal(range(len(xon_attrs)), idx_attr)
            # print("Trying to remove pair (attr=%d, thresh=%.5f)..." % (last_xon.idx_attr[idx_attr],
            #                                                            last_xon.thresh_val[idx_attr]))

            valid_attrs = np.compress(mask, last_xon.idx_attr)
            valid_thresh = np.compress(mask, last_xon.thresh_val)
            new_xon_vals = _apply_attr(train_feats=train_feats,
                                       valid_attrs=valid_attrs,
                                       valid_thresh=valid_thresh)
            new_xon_thresh = np.unique(new_xon_vals)
            # returns: best_gini, idx_best_thresh
            best_gini, idx_best_thresh = _res_gini_numerical(feat=new_xon_vals,
                                                             target=train_labels,
                                                             sorted_thresholds=new_xon_thresh)

            # print("Current gini is: %.5f..." % best_gini)
            new_cost = _eval_attr(curr_gini=best_gini,
                                  best_gini=ovr_best_gini,
                                  best_compl=ovr_best_compl,
                                  train_feats=train_feats,
                                  attr_feats=valid_attrs,
                                  attr_thresh=valid_thresh,
                                  available_attrs=available_attrs)

            if new_cost:
                ovr_best_gini = best_gini
                split_val = new_xon_thresh[idx_best_thresh]
                idx_best_attr = idx_attr
                ovr_best_compl = new_cost

        # print("After trying to delete an (attr, val) pair, best gini obtained was: %.5f..." % ovr_best_gini)

        if ovr_best_gini < prior_gini or ovr_best_compl < last_xon.cost:
            # construct new X-of-N attribute by deleting `xon_attrs[idx_best_attr]` and corresponding thresh
            mask = np.not_equal(range(len(xon_attrs)), idx_best_attr)
            new_attr = XOfNAttribute(idx_attr=np.compress(mask, last_xon.idx_attr).tolist(),
                                     thresh_val=np.compress(mask, last_xon.thresh_val).tolist(),
                                     split_val=split_val,
                                     cost=ovr_best_compl)

    else:
        for i, idx_attr in enumerate(available_attrs):
            valid_attrs = np.array(last_xon.idx_attr + [idx_attr])
            curr_attr_thresh = _find_valid_values(train_feats[:, idx_attr], train_labels) \
                if available_thresh is None else available_thresh[i]
            for thr in curr_attr_thresh:
                valid_thresh = np.array(last_xon.thresh_val + [thr])

                new_xon_vals = np.sum(train_feats[:, valid_attrs] < valid_thresh, axis=1)
                new_xon_thresh = np.unique(new_xon_vals)
                best_gini, idx_best_thresh = _res_gini_numerical(feat=new_xon_vals,
                                                                 target=train_labels,
                                                                 sorted_thresholds=new_xon_thresh)

                # print("Current gini is: %.5f..." % best_gini)
                new_cost = _eval_attr(curr_gini=best_gini,
                                      best_gini=ovr_best_gini,
                                      best_compl=ovr_best_compl,
                                      train_feats=train_feats,
                                      attr_feats=valid_attrs,
                                      attr_thresh=valid_thresh,
                                      available_attrs=available_attrs)

                if new_cost:
                    ovr_best_gini = best_gini
                    split_val = new_xon_thresh[idx_best_thresh]
                    ovr_best_thresh = thr
                    ovr_best_compl = new_cost
                    idx_best_attr = idx_attr

        if ovr_best_gini < prior_gini or ovr_best_compl < last_xon.cost:
            # construct new X-of-N attribute by adding (attr, val) pair which resulted in best gini value (< prior_gini)
            new_attr = XOfNAttribute(idx_attr=(last_xon.idx_attr + [idx_best_attr]),
                                     thresh_val=(last_xon.thresh_val + [ovr_best_thresh]),
                                     split_val=split_val,
                                     cost=ovr_best_compl)

    return new_attr, ovr_best_gini


def very_greedy_construct_xofn(train_feats, train_labels, available_attrs=None, available_thresh=None):
    """WARNING: experimental!"""
    if train_feats.ndim == 0:
        train_feats = np.expand_dims(train_feats, 0)

    if available_attrs is None:
        available_attrs = np.arange(train_feats.shape[1])

    # element at index i is best XofN attribute that consists of i attributes
    best_xons = [None]
    del_applied = [True]

    best_gini, best_thresh, idx_best_attr = 1 + 0.01, np.nan, 0
    best_compl = np.inf
    # `attr_best_thresh[i]` is the best threshold for attribute `available_attrs[i]`
    attr_best_thresh = []

    for i, idx_attr in enumerate(available_attrs):
        curr_thresh = _find_valid_values(train_feats[:, idx_attr], train_labels) if available_thresh is None else \
            available_thresh[i]
        gini, idx_thresh = _res_gini_numerical(feat=train_feats[:, idx_attr],
                                               target=train_labels,
                                               sorted_thresholds=curr_thresh)
        attr_best_thresh.append([curr_thresh[idx_thresh]])

        new_cost = _eval_attr(curr_gini=gini,
                              best_gini=best_gini,
                              best_compl=best_compl,
                              train_feats=train_feats,
                              attr_feats=[idx_attr],
                              attr_thresh=curr_thresh[idx_thresh],
                              available_attrs=available_attrs)

        if new_cost:
            best_gini = gini
            best_thresh = curr_thresh[idx_thresh]
            idx_best_attr = idx_attr
            best_compl = new_cost

    best_xons.append(XOfNAttribute([idx_best_attr], [best_thresh], split_val=1, cost=best_compl))
    del_applied.append(True)  # deletion attempt would be pointless

    # length of last X-of-N attribute constructed
    len_last_xon = 1
    # number of consequent iterations in which no insertion of new (attr, val) pairs was performed
    iters_no_add = 0

    while len_last_xon > 0:
        if iters_no_add == 5:
            break
        # specifies if the algorithm should try deletion or insertion of an (attr, val) pair
        do_del = not del_applied[len_last_xon]
        # print("Trying to %s an (attr, val) %s X-of-N of size %d..." % ("delete" if do_del else "insert",
        #                                                                "from" if do_del else "into", len_last_xon))
        new_attr, new_gini = search_xofn(train_feats=train_feats,
                                         train_labels=train_labels,
                                         available_attrs=available_attrs,
                                         available_thresh=attr_best_thresh,
                                         last_xon=best_xons[len_last_xon],
                                         op_del=do_del)

        if do_del:
            del_applied[len_last_xon] = True

            if new_attr is not None:
                # delete resulted in a better X-of-N attribute
                best_gini = new_gini
                len_last_xon -= 1
                best_xons[len_last_xon] = new_attr
                if len_last_xon > 1:
                    del_applied[len_last_xon] = False
                iters_no_add += 1
        else:
            if new_attr is None:
                # tried both deletion and insertion on current attribute, nothing resulted in a better attribute
                # print("Neither DEL nor INS produced better attribute, ending... [Best X-of-N attribute length: %d]"
                #       % len_last_xon)
                break
            else:
                iters_no_add = 0

                best_gini = new_gini
                len_last_xon += 1
                if len_last_xon >= len(best_xons):
                    best_xons.append(new_attr)
                    del_applied.append(False)
                else:
                    best_xons[len_last_xon] = new_attr
                    # del_applied[len_last_xon] = False

    return best_xons[len_last_xon], best_gini


def construct_xofn(train_feats, train_labels, available_attrs=None):
    """ In a greedy way constructs an X-of-N attribute that has the best trade-off between:
    (1) minimizing gini index and
    (2) minimizing attribute complexity.

    Parameters
    ----------
    train_feats: np.array

    train_labels: np.array

    available_attrs: list or np.array, optional
        Specifies attributes (indices of columns that they belong to in `train_feats`) that should be taken into
        account when constructing X-of-N attribute. One case where this is useful is for constructing random forests.
        If not specified, use all attributes of `train_feats`.

    Returns
    -------
    (XOfNAttribute, float)
        First attribute represents newly constructed X-of-N attribute (may consist of just 1 primitive attribute and
        corresponding threshold) and the second represents gini index obtained with new X-of-N attribute.
    """
    if train_feats.ndim == 0:
        train_feats = np.expand_dims(train_feats, 0)

    if available_attrs is None:
        available_attrs = np.arange(train_feats.shape[1])

    # element at index i is best XofN attribute that consists of i attributes
    best_xons = [None]
    del_applied = [True]

    best_gini, best_thresh, idx_best_attr = 1 + 0.01, np.nan, 0
    best_compl = np.inf

    # find best length 1 X-of-N attribute separately
    for idx_attr in available_attrs:
        curr_thresh = _find_valid_values(train_feats[:, idx_attr], train_labels)
        gini, idx_thresh = _res_gini_numerical(feat=train_feats[:, idx_attr],
                                               target=train_labels,
                                               sorted_thresholds=curr_thresh)
        new_cost = _eval_attr(curr_gini=gini,
                              best_gini=best_gini,
                              best_compl=best_compl,
                              train_feats=train_feats,
                              attr_feats=[idx_attr],
                              attr_thresh=curr_thresh[idx_thresh],
                              available_attrs=available_attrs)

        if new_cost:
            best_gini = gini
            best_thresh = curr_thresh[idx_thresh]
            idx_best_attr = idx_attr
            best_compl = new_cost

    best_xons.append(XOfNAttribute([idx_best_attr], [best_thresh], split_val=1, cost=best_compl))
    del_applied.append(True)  # deletion attempt would be pointless

    # length of last X-of-N attribute constructed
    len_last_xon = 1
    # number of consequent iterations in which no insertion of new (attr, val) pairs was performed
    iters_no_add = 0

    while len_last_xon > 0:
        if iters_no_add == 5:
            break
        # specifies if the algorithm should try deletion or insertion of an (attr, val) pair
        do_del = not del_applied[len_last_xon]
        # print("Trying to %s an (attr, val) %s X-of-N of size %d..." % ("delete" if do_del else "insert",
        #                                                                "from" if do_del else "into", len_last_xon))
        new_attr, new_gini = search_xofn(train_feats=train_feats,
                                         train_labels=train_labels,
                                         available_attrs=available_attrs,
                                         last_xon=best_xons[len_last_xon],
                                         op_del=do_del)

        if do_del:
            del_applied[len_last_xon] = True

            if new_attr is not None:
                # delete resulted in a better X-of-N attribute
                best_gini = new_gini
                len_last_xon -= 1
                best_xons[len_last_xon] = new_attr
                if len_last_xon > 1:
                    del_applied[len_last_xon] = False
                iters_no_add += 1
        else:
            if new_attr is None:
                # tried both deletion and insertion on current attribute, nothing resulted in a better attribute
                # print("Neither DEL nor INS produced better attribute, ending... [Best X-of-N attribute length: %d]"
                #       % len_last_xon)
                break
            else:
                iters_no_add = 0

                best_gini = new_gini
                len_last_xon += 1
                if len_last_xon >= len(best_xons):
                    best_xons.append(new_attr)
                    del_applied.append(False)
                else:
                    best_xons[len_last_xon] = new_attr
                    # del_applied[len_last_xon] = False

    return best_xons[len_last_xon], best_gini


class TreeNode(object):
    __slots__ = ('attr_list', 'thresh_list', 'split_val', 'is_leaf', 'outcome', 'probas', 'lch', 'rch')

    def __init__(self, is_leaf):
        """
        Parameters
        ----------
        is_leaf: bool
            Specifies whether node is internal (splits data) or a leaf (contains outcome)

        Notes
        -----
            Create internal/leaf nodes using static methods TreeNode.create_leaf() and TreeNode.create_internal().
        """
        self.attr_list = None
        self.thresh_list = None
        self.split_val = None
        self.is_leaf = is_leaf
        self.outcome = None
        self.probas = None
        self.lch = None
        self.rch = None

    @staticmethod
    def create_leaf(probas, outcome):
        node = TreeNode(is_leaf=True)
        node.outcome = outcome
        node.probas = probas

        return node

    @staticmethod
    def create_internal(attr_list, thresh_list, split_val, lch=None, rch=None):
        node = TreeNode(is_leaf=False)
        node.attr_list = attr_list
        node.thresh_list = thresh_list
        node.split_val = split_val
        node.lch = lch
        node.rch = rch

        return node


class XOfNTree(object):
    __slots__ = ('min_samples_leaf', 'max_features', 'max_depth', 'classes_', '_max_feats', '_min_samples',
                 'labels_encoded', '_root', '_is_fitted')

    def __init__(self, min_samples_leaf=1,
                 max_features=None,
                 max_depth=None,
                 random_state=None,
                 labels_encoded=False,
                 classes_=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth if max_depth is not None else 2 ** 30
        self.labels_encoded = labels_encoded
        if random_state is not None:
            np.random.seed(random_state)

        self.classes_ = classes_
        self._max_feats = None
        self._min_samples = None
        self._root = None
        self._is_fitted = False

    @staticmethod
    def calc_max_feats(state, n_feats):
        if state is None:
            return n_feats
        elif isinstance(state, int):
            return state
        elif isinstance(state, float):
            return int(state * n_feats)
        elif state in ("auto", "sqrt"):
            return int(np.sqrt(n_feats))
        elif state == "log2":
            return int(np.log2(n_feats))
        else:
            raise ValueError("Invalid 'max_features' value encountered (%s)..." % str(state))

    @staticmethod
    def calc_min_samples(state, n_samples):
        if isinstance(state, int):
            return state
        elif isinstance(state, float):
            return int(state * n_samples)
        else:
            raise ValueError("Invalid 'min_samples_leaf' value encountered (%s)..." % str(state))

    def encode_labels(self, labels):
        self.classes_, enc_labels = np.unique(labels, return_inverse=True)
        return enc_labels

    def fit(self, train_feats, train_labels):
        self._is_fitted = False
        if train_feats.ndim == 1:
            train_feats = np.expand_dims(train_feats, 0)

        if not self.labels_encoded:
            train_labels = self.encode_labels(train_labels)
        self._max_feats = XOfNTree.calc_max_feats(self.max_features, train_feats.shape[1])
        self._min_samples = XOfNTree.calc_min_samples(self.min_samples_leaf, train_feats.shape[0])

        self._root = self._split_rec(train_feats, train_labels, 0)
        self._is_fitted = True

    def _split_rec(self, curr_feats, curr_labels, curr_depth):
        # Note: `curr_feats` should be 2-dim (check is handled in self.fit())
        n_attrs = curr_feats.shape[1]
        n_samples = curr_feats.shape[0]
        uniqs, class_dist = np.unique(curr_labels, return_counts=True)

        if curr_depth == self.max_depth:
            probas = np.zeros_like(self.classes_, dtype=np.float32)
            probas[uniqs] = class_dist / n_samples
            return TreeNode.create_leaf(probas, outcome=np.argmax(probas))

        if class_dist.shape[0] == 1:
            # pure subset
            probas = np.zeros_like(self.classes_, dtype=np.float32)
            probas[curr_labels[0]] = 1
            return TreeNode.create_leaf(probas, outcome=curr_labels[0])

        prior_gini = _gini(class_dist, n_samples)
        selected_attrs = np.random.choice(n_attrs, size=self._max_feats, replace=False) \
            if self._max_feats < n_attrs else np.arange(n_attrs)
        new_attr, new_gini = very_greedy_construct_xofn(curr_feats, curr_labels, available_attrs=selected_attrs)

        # if best possible constructed attribute is of length > 1 and has same gini, that means it reduces
        # representation complexity (which means the algorithm should not terminate just yet)
        if (len(new_attr) == 1 and new_gini >= prior_gini) or (len(new_attr) > 1 and new_gini > prior_gini):
            probas = np.zeros_like(self.classes_, dtype=np.float32)
            probas[uniqs] = class_dist / n_samples
            return TreeNode.create_leaf(probas, outcome=np.argmax(probas))

        # print("Prior: %.5f vs new: %.5f" % (prior_gini, new_gini))
        # contains number of true conditions in newly created X-of-N attribute for each row in `train_feats`
        xon_vals = _apply_attr(curr_feats,
                               valid_attrs=new_attr.idx_attr,
                               valid_thresh=new_attr.thresh_val)

        node = TreeNode.create_internal(attr_list=new_attr.idx_attr,
                                        thresh_list=new_attr.thresh_val,
                                        split_val=new_attr.split_val)

        lch_mask = xon_vals < new_attr.split_val
        rch_mask = np.logical_not(lch_mask)

        lfeats, llabs = curr_feats[lch_mask, :], curr_labels[lch_mask]
        rfeats, rlabs = curr_feats[rch_mask, :], curr_labels[rch_mask]

        if llabs.shape[0] < self._min_samples or rlabs.shape[0] < self._min_samples:
            # further split would result in a node having to learn on a subset that is too small
            probas = np.zeros_like(self.classes_, dtype=np.float32)
            probas[uniqs] = class_dist / n_samples
            return TreeNode.create_leaf(probas, outcome=np.argmax(probas))

        node.lch = self._split_rec(lfeats, llabs, curr_depth + 1)
        node.rch = self._split_rec(rfeats, rlabs, curr_depth + 1)
        return node

    def predict(self, test_feats):
        return self.classes_[np.argmax(self.predict_proba(test_feats), axis=1)]

    def predict_proba(self, test_feats):
        """
        Parameters
        ----------
        test_feats: np.array

        Returns
        -------
        np.array
            Predicted probabilities for each example in `test_feats`. Probabilities are placed in the order, specified
            by `self.classes_`.
        """
        if not self._is_fitted:
            raise Exception("Model not fitted! Please call fit() first...")

        if test_feats.ndim == 1:
            test_feats = np.expand_dims(test_feats, 0)

        n_samples = test_feats.shape[0]
        return np.array([self._single_pred_proba(test_feats[idx_ex, :], self._root)
                        for idx_ex in range(n_samples)])

    def _single_pred_proba(self, single_example, curr_node):
        """
        Parameters
        ----------
        single_example: np.array
            Example for which prediction will be made.
        curr_node: TreeNode

        Returns
        -------
        np.array
            Probability predictions for `single_example`.
        """

        if curr_node.is_leaf:
            return curr_node.probas

        xon_value = _apply_attr(single_example, curr_node.attr_list, curr_node.thresh_list)

        if xon_value < curr_node.split_val:
            return self._single_pred_proba(single_example, curr_node.lch)
        else:
            return self._single_pred_proba(single_example, curr_node.rch)


class RandomXOfNForest(object):
    def __init__(self, n_estimators=100,
                 min_samples_leaf=1,
                 max_features="sqrt",
                 sample_size=None,
                 max_depth=None,
                 random_state=None,
                 labels_encoded=False,
                 classes_=None):
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.sample_size = sample_size
        self.max_depth = max_depth
        if random_state is not None:
            np.random.seed(random_state)
        self.labels_encoded = labels_encoded
        self.classes_ = classes_

        self.estimators = []
        self._is_fitted = False
        self._sample_size = None

    @staticmethod
    def calc_sample_size(state, n_samples):
        if state is None:
            return n_samples
        elif isinstance(state, float):
            return int(state * n_samples)
        elif isinstance(state, int):
            return state
        else:
            raise ValueError("Invalid 'sample_size' value encountered (%s)..." % str(state))

    def encode_labels(self, labels):
        self.classes_, enc_labels = np.unique(labels, return_inverse=True)
        return enc_labels

    def fit(self, train_feats, train_labels):
        self._is_fitted = False
        self.estimators = []

        if train_feats.ndim == 1:
            train_feats = np.expand_dims(train_feats, 0)

        if not self.labels_encoded:
            train_labels = self.encode_labels(train_labels)

        n_samples = train_feats.shape[0]
        self._sample_size = RandomXOfNForest.calc_sample_size(self.sample_size, n_samples)
        # print("Using bootstrap sample of size %d..." % self._sample_size)

        for i in range(self.n_estimators):
            # print("Training tree #%d" % i)
            sample_idx = np.random.choice(n_samples, size=self._sample_size, replace=True)
            curr_estimator = XOfNTree(min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features,
                                      max_depth=self.max_depth,
                                      labels_encoded=True,
                                      classes_=self.classes_)
            curr_estimator.fit(train_feats[sample_idx, :], train_labels[sample_idx])

            self.estimators.append(curr_estimator)

        self._is_fitted = True

    def predict_proba(self, test_feats):
        if not self._is_fitted:
            raise Exception("Model not fitted! Please call fit() first...")
        if test_feats.ndim == 1:
            test_feats = np.expand_dims(test_feats, 0)

        n_samples = test_feats.shape[0]
        proba_preds = np.zeros((n_samples, self.classes_.shape[0]), dtype=np.float32)

        for i in range(self.n_estimators):
            preds = self.estimators[i].predict_proba(test_feats)
            proba_preds += preds

        proba_preds = np.divide(proba_preds, self.n_estimators)
        return proba_preds

    def predict(self, test_feats):
        return self.classes_[np.argmax(self.predict_proba(test_feats), axis=1)]
