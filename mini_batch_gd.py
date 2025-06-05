from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
import time

X,y = make_regression(n_samples=500,n_features=1,n_targets=1,n_informative=1,noise=50,random_state=42)

class Mini_batch_GD:
    def __init__(self,epochs,learing_rate):
        self.m = -345
        self.b = 435
        self.epochs = epochs
        self.learning_rate = learing_rate
    
    def batch_generator(self,idx,X,y,batch_size):
        X_batch = []
        y_batch = []
        for start_idx in range(0,len(X),batch_size):
            end_idx = start_idx + batch_size
            batch_idx = idx[start_idx:end_idx]
            X_batch.append(X[batch_idx]) 
            y_batch.append(y[batch_idx])
        return X_batch,y_batch

    def fit(self,X,y):
        st = time.time()
        for _ in range(self.epochs):
            idx = np.arange(len(X))
            np.random.shuffle(idx)
            X_batch,y_batch = self.batch_generator(idx,X,y,batch_size=25)
            
            for i in range(len(X_batch)):
                dm = -2 * np.sum((y_batch[i] - (self.m*X_batch[i].ravel() + self.b))*X_batch[i].ravel())/len(X_batch[0])
                db = -2 * np.sum((y_batch[i] - (self.m*X_batch[i].ravel() + self.b)))/len(X_batch[0])

                self.m -= self.learning_rate*dm
                self.b -= self.learning_rate*db
        end = time.time()
        print('weight : ',self.m)
        print('bias :',self.b)
        print('time taken :',end-st)
        return self.m,self.b
    
    
    
    def plot(self,X,y):
        plt.scatter(X,y)
        plt.plot(X,self.m*X+self.b,'r')
        plt.title('Best fit line')
        plt.show()
        
lr = Mini_batch_GD(20,0.01)
lr.fit(X,y)
lr.plot(X,y)
