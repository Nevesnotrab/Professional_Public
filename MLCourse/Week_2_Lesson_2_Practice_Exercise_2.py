"""Practice Problem 2: Implement the core splitting logic for a decision tree
    classifier."""

import numpy as np
from collections import Counter
#from sklearn.datasets import load_files
from sklearn.datasets import load_iris

#TASKS
# 1. Implement functions:
#   a. gini_impurity(y) - calculates Gini impurity
#   b. entropy(y) - Calculates entropy
#   c. information_gain(y, y_left, y_right) - calculates information gain
# 2. Create a TreeNode class that stores:
#   Feature index and threshold (for internal nodes)
#   Class prediction (for leaf nodes)
#   Left and right children
# 3. Implement build_tree(X, y, max_depth=3) that recursively builds the tree
# 4. Test your splitting logic on the iris dataset and verify that it makes
#   reasonable splits.


# 1 a) Gini Impurity
def GiniImpurity(D, mode = "probability"):
    def ProbabilityGini(p):
        Gini = 1 - np.sum(p**2)
        return Gini
    
    def CountGini(counts):
        p = counts / np.sum(counts)
        return ProbabilityGini(p)

    if mode == "probability":
        if isinstance(D, np.ndarray) and D.ndim == 1:
            if not np.isclose(np.sum(D), 1.0):
                print("Probabilities must sum to 1")
                return None
            return ProbabilityGini(D)
        else:
            print("Probabilities must be a 1D numpy array.")
            return None

    if mode == "count":
        if isinstance(D, np.ndarray) and D.ndim == 1:
            if np.issubdtype(D.dtype, np.integer):
                return CountGini(D)
            else:
                print("Count array must contain integers.")
                return None
        else:
            print("Count must be a 1D numpy array.")
            return None

    elif mode == "raw":
        if isinstance(D, np.ndarray) and D.ndim == 1:
            labels, counts = np.unique(D, return_counts=True)
            return CountGini(counts)
        else:
            print("Raw data must be a 1D numpy array of labels.")
            return None
    
    print("Unrecognized mode.")
    return None

# 1 b) Entropy
def Entropy(X):
    #Requires 1-D array
    if len(np.shape(X)) != 1:
        print("Array must be 1D")
        return
    entropy = 0
    for i in range(len(X)):
        entropy += X[i]*np.log2(X[i])
    entropy = entropy*-1
    return entropy

# 1 c) Information Gain
def InformationGain(y, y_left, y_right):
    if np.ndim(y) != 1 or np.ndim(y_left) != 1 or np.ndim(y_right) != 1:
        print("All lists must be 1D")
        return

    unique_entries, counts = np.unique(y, return_counts=True)
    tot_entries = len(y)

    unique_entries_left, counts_left = np.unique(y_left, return_counts=True)
    tot_left = len(y_left)

    unique_entries_right, counts_right = np.unique(y_right, return_counts=True)
    tot_right = len(y_right)

    #Calculate parent node entropy:
    p_parent = counts / tot_entries
    Hp = Entropy(p_parent)

    #Calculate left entropy:
    p_left = counts_left / tot_left
    HL = Entropy(p_left)

    #Calculate right entropy:
    p_right = counts_right / tot_right
    HR = Entropy(p_right)

    #Calculate average entropy:
    HLR = (tot_left/tot_entries)*HL + (tot_right/tot_entries)*HR

    IG = Hp - HLR
    return(IG)


a = np.array(['a', 'a', 'b', 'b', 'a', 'c', 'a', 'b', 'b', 'b'])
a_left = np.array(['a', 'a', 'b', 'c'])
a_right = np.array(['b', 'a', 'a', 'b'])

#IG = InformationGain(a, a_left, a_right)
#print(IG)

# 2. Create a TreeNode class that stores:
#   Feature index and threshold (for internal nodes)
#   Class prediction (for leaf nodes)
#   Left and right children

class TreeNode():
    def __init__(self, node_feature_index=None, threshold=None,\
                 left_child=None, right_child=None, class_label=None):
        self.feature_index = node_feature_index     # Internal nodes only
        self.threshold = threshold                  # Internal nodes only
        self.left_child = left_child
        self.right_child = right_child
        self.class_label = class_label              # Leaf nodes only

    def is_leaf(self):
        return self.class_label is not None

    def route(self, X_sample, sample_index):
        if X_sample[sample_index][self.feature_index] < self.threshold:
            return 0 #0 for left child
        else:
            return 1 #1 for right child
        
    def predict(self, X_sample):
        if self.is_leaf():
            return self.class_label
        if X_sample[self.feature_index] < self.threshold:
            return self.left_child.predict(X_sample)
        else:
            return self.right_child.predict(X_sample)

# 3. Implement build_tree(X, y, max_depth=3) that recursively builds the tree
def build_tree(X, y, max_depth=3):
    # Check if all y values are the same:
    unique_entries, counts = np.unique(y, return_counts=True)
    if len(counts) == 1:
        leaf_node = TreeNode(class_label=y[0])
        return leaf_node
    # If max_depth is 1, this is a recursive call where we've called it down to
    #   the last depth. Make a leaf node with the most common class.
    if max_depth == 1:
        leaf_node = TreeNode(class_label=y[np.argmax(counts)])
        return leaf_node
    if len(y) == 0:
        leaf_node = TreeNode(class_label="None")
        return leaf_node

    # Find the best split
    # a. Initialize tracking variables
    lowest_wa_gini = None #Lowest weighted average Gini impurity
    f = None # feature index
    t = None # threshold

    # b. Loop through all features
    for feature_index in range(X.shape[1]):
        feature_values = X[:,feature_index] # Get feature columns

        # c. Sort Feature Values
        sorted_indices = np.argsort(feature_values)
        feature_values = feature_values[sorted_indices]
        y_sorted = y[sorted_indices]

        # d. Enumerate Potential Thresholds
        for i in range(1, len(feature_values)):
            if feature_values[i] == feature_values[i-1]:
                continue
            threshold = (feature_values[i] + feature_values[i-1]) / 2

            # e. Split y into Left and Right Based on Threshold
            y_left = y_sorted[:i]
            y_right = y_sorted[i:]

            gini_left = GiniImpurity(y_left, mode="raw")
            gini_right = GiniImpurity(y_right, mode="raw")

            # f. Compute weighted averages
            n = len(y)
            n_left = len(y_left)
            n_right = len(y_right)

            weighted_gini = (n_left/n) * gini_left + (n_right/n) * gini_right

            if (lowest_wa_gini is None) or (weighted_gini < lowest_wa_gini):
                lowest_wa_gini = weighted_gini
                f = feature_index
                t = threshold
    
    if f is None or len(y_left) == 0 or len(y_right) == 0:
        leaf_node = TreeNode(class_label=unique_entries[np.argmax(counts)])
        return leaf_node

    # Split the data and Recurse:
    mask_left = X[:,f] < t
    mask_right = ~mask_left

    X_left = X[mask_left]
    y_left = y[mask_left]
    X_right = X[mask_right]
    y_right = y[mask_right]

    left_node = build_tree(X_left, y_left, max_depth - 1)
    right_node = build_tree(X_right, y_right, max_depth - 1)

    NewNode = TreeNode(f, t, left_node, right_node)

    # Return the TreeNode
    return NewNode


"""
X = np.array([
    [2.5],
    [3.0],
    [4.5], 
    [6.0]
])

y = np.array([0, 0, 1, 1])

tree = build_tree(X, y, max_depth=2)
prediction = tree.predict(np.array([3.5]))
print(prediction)
"""

iris = load_iris()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris.data,\
                                                    iris.target,\
                                                    test_size=0.2,\
                                                    random_state=42)


tree = build_tree(X_train, y_train, max_depth=3)
predictions = [tree.predict(sample) for sample in X_test]
predictions = np.int_(predictions)
print(predictions)
print(y_test)

num_wrong = 0
n = len(y_test)
wrong_indices = []
for i in range(n):
    if predictions[i] != y_test[i]:
        num_wrong+=1
        wrong_indices.append(i)

print("Total wrong predictions: ", num_wrong)
print("Success percentage: ", (n-num_wrong)/n)
print("Wrong indices: ", wrong_indices)