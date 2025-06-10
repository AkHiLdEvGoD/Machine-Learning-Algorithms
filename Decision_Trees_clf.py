import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X,y = load_iris(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

class Node:
    def __init__(self,feature=None,thresh=None,right=None,left=None,*,value=None):
        self.feature_index = feature
        self.thresh = thresh
        self.right_tree = right
        self.left_tree = left
        self.value = value        

    def is_leaf(self):
        return self.value is not None

class Decision_Tree_Classifier:
    def __init__(self,max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self,X,y):
        self.root = self.grow_tree(X,y)
    
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
                
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                
                left_node_gini = self.gini(y[left_mask])
                right_node_gini = self.gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) * left_node_gini + len(y[right_mask]) * right_node_gini)/len(X)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = thresh
        
        return best_feature,best_threshold
    
    def grow_tree(self,X,y,depth=0):
        classes = np.unique(y)
        
        print(f"\n At depth {depth}:")
        print(f"  Classes: {np.unique(y)}")
        print(f"  max_depth: {self.max_depth}")

        if depth >= self.max_depth or len(classes) == 1 :
            leaf_value = self.majority_class(y)
            print(f"Stopping: returning leaf with value = {leaf_value}")
            return Node(value = leaf_value)
        
        feature,thresh = self.best_split(X,y)
        print(f"  Best Split -> Feature: {feature}, Threshold: {thresh}")
        
        if feature is None:
            print(f"  ⚠️ No valid split found. Returning leaf with value = {leaf_value}")
            return Node(value=self.majority_class(y))
        
        left_mask = X[:,feature] <= thresh
        right_mask = X[:,feature] > thresh

        left_tree = self.grow_tree(X[left_mask],y[left_mask],depth+1)
        right_tree = self.grow_tree(X[right_mask],y[right_mask],depth+1)
        
        return Node(feature = feature,thresh = thresh,right=right_tree,left=left_tree)
    
    def majority_class(self,y):
        values,counts = np.unique(y,return_counts=True)
        return values[np.argmax(counts)]

    def predict(self,X):
        return np.array([self._predict(input,self.root) for input in X])

    def _predict(self,input,node):
        if node.is_leaf():
            return node.value
        if input[node.feature_index] <= node.thresh:
            return self._predict(input,node.left_tree)
        else:
            return self._predict(input,node.right_tree)

tree = Decision_Tree_Classifier(4)
tree.fit(X_train,y_train)
y_pred = tree.predict(X_test)
print('Accuarcy :',accuracy_score(y_pred,y_test))
