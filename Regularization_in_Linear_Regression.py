import numpy as np
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X,y = make_regression(n_samples=1000,n_features=10,n_targets=1,n_informative=1,noise=10,random_state=42)

class Regularizaton:
    def __init__(self,epochs: int,learning_rate: float,regularization: str,lamda: float):
        self.epochs=epochs
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.lamda = lamda
    
    def fit(self,X,y):
        X = np.insert(X,0,1,axis=1)
        self.weights = np.random.rand(X.shape[1])
        for _ in range(self.epochs):
            y_pred = np.dot(X,self.weights)
            error = y - y_pred

            if self.regularization == 'L2' : 
                reg_term = self.lamda * self.weights
                reg_term[0] = 0
                gradient = (-np.dot(X.T,error) / len(X)) + reg_term

            elif self.regularization == 'L1' : 
                reg_term = self.lamda * np.sign(self.weights)
                reg_term[0] = 0
                gradient = (-np.dot(X.T,error) / len(X)) + reg_term
            
            else:
                gradient = (-np.dot(X.T,error) / len(X))
            
            self.weights -= self.learning_rate * gradient
        training_error = mean_squared_error(y,np.dot(X,self.weights))
        print('Weights :',self.weights[1:])
        print('Bias :',self.weights[0])
        print('Training error :',training_error)
        return self.weights

    def predict(self,X):
        X = np.insert(X,0,1,axis=1)
        return np.dot(X,self.weights)


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
reg = Regularizaton(1000,0.01,'L1',0.5)
reg.fit(X_train,y_train)
y_pred = reg.predict(X_test)

print('Test error',mean_squared_error(y_pred,y_test))


