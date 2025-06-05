import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import accuracy_score

iris = load_iris()
X,y = iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

class knn:
    def __init__(self,k,dist_metric):
        self.k = k
        self.dist_metric = dist_metric

    def manhattan_dist(self,x1,x2):
        return np.abs(np.sum((x1-x2)))
    
    def eucledian_dist(self,x1,x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X_test):
        pred = []
        for x in X_test:
            if self.dist_metric == 'Eucledian':
                distance = [self.eucledian_dist(x,x_train) for x_train in self.X_train]
            else:
                distance = [self.manhattan_dist(x,x_train) for x_train in self.X_train]
            
            kth_indices = np.argsort(distance)[:self.k]
            y_label = [self.y_train[i] for i in kth_indices]

            most_common = Counter(y_label).most_common(1)[0][0]
            pred.append(most_common)
        return pred

    def accuracy(self,X,y):
        y_pred = self.predict(X)
        print('Accuracy :',accuracy_score(y_pred,y))
        print(y)
        print(y_pred)

knn_clf = knn(9,'Eucledian')
knn_clf.fit(X_train,y_train)
knn_clf.accuracy(X_test,y_test)