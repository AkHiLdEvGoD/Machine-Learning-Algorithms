import numpy as np
from sklearn.datasets import load_iris

X,y = load_iris(return_X_y=True)

class Decision_Tree_Classifier:
    def __init__(self,max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self,X,y):
        self.root = self.tree(X,y)
    
    def gini(self,y):
        _,counts = np.unique(y,return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def best_split(self,X,y):
        n_features = len(X[0])
        best_gini = 1                                       # For multiclass datasets
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            threshold = np.unique(X[:,feature])
            for thresh in threshold:

                left_mask = X[:,feature] <= thresh
                right_mask = X[:,feature] > thresh
                
                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue
                
                left_node_gini = self.gini(y[left_mask])
                right_node_gini = self.gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) * left_node_gini + len(y[right_mask]) * right_node_gini)/len(X)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = thresh
        
        print(best_feature)
        print(best_threshold)
        print(best_gini)
        return best_feature,best_threshold

tree = Decision_Tree_Classifier(5)
tree.best_split(X,y)