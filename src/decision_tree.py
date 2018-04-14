class InternalTreeNode:
    def __init__(self):
        self.split_attr_idx, self.split_val = None, None

# contains class prediction
class LeafTreeNode:
    def __init__(self):
        # {class: probability}
        self.probabilities = {}
        self.outcome = None