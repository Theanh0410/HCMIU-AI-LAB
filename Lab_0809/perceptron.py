import numpy as np

class MultiClassPerceptron:
    def __init__(self, n_classes, learning_rate=0.01, max_iter=1000):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.biases = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.n_classes, n_features))  # One weight vector per class
        self.biases = np.zeros(self.n_classes)  # One bias per class

        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                scores = np.dot(self.weights, x_i) + self.biases
                predicted_class = np.argmax(scores)
                
                if predicted_class != y[idx]:
                    self.weights[y[idx]] += self.learning_rate * x_i  # Correct class
                    self.biases[y[idx]] += self.learning_rate
                    self.weights[predicted_class] -= self.learning_rate * x_i  # Incorrect class
                    self.biases[predicted_class] -= self.learning_rate

    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)