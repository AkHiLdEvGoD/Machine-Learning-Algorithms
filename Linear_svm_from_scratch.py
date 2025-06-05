import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)

class svm:
    def __init__(self,epochs,lamda,learning_rate):
        self.epochs = epochs
        self.lamda = lamda
        self.learning_rate = learning_rate
        
    def fit(self,X,y):
        idx = np.arange(X.shape[0])
        self.weights = np.random.rand(X.shape[1]) * 0.01
        self.bias = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for i in idx:
                condition = y[i]*(np.dot(self.weights.T,X[i]) + self.bias) >= 1 
                if condition :
                    dw = self.lamda * self.weights
                    db = 0
                else :
                    dw = self.lamda * self.weights - y[i] * X[i]
                    db = -y[i]
                self.weights -= self.learning_rate*dw
                self.bias -= self.learning_rate*db
        print('Weights :',self.weights)
        print('bias :',self.bias)
        
        return self.weights,self.bias

    def visualize(self, X, y):
        plt.figure(figsize=(8, 6))

        # Plot data points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30, edgecolors='k')

        # Draw decision boundary and margins
        ax = plt.gca()
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
        w = self.weights
        b = self.bias
        y_vals = -(w[0] * x_vals + b) / w[1]

        # Margins: y = -(w.x + b ± 1)/w[1]
        margin = 1 / np.linalg.norm(w)
        y_vals_plus = y_vals + margin
        y_vals_minus = y_vals - margin

        plt.plot(x_vals, y_vals, 'k-')  # decision boundary
        plt.plot(x_vals, y_vals_plus, 'k--')
        plt.plot(x_vals, y_vals_minus, 'k--')

        # Approximate support vectors: margin ≈ 1
        distances = y * (np.dot(X, w) + b)
        support_vector_indices = np.where((distances >= 0.99) & (distances <= 1.01))[0]
        plt.scatter(X[support_vector_indices][:, 0], X[support_vector_indices][:, 1],
                    s=100, linewidth=1, facecolors='none', edgecolors='g', label='Support Vectors')

        plt.legend()
        plt.title("SVM Decision Boundary with Support Vectors")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)
        plt.show()

svm_clf = svm(500,0.01,0.01)
svm_clf.fit(X,y)
svm_clf.visualize(X,y)
# plt.scatter(X[:,0],X[:,1])
# plt.show()
# print(y)
# print(X.shape[1])
# print((np.random.rand(X.shape[1])).T.shape)