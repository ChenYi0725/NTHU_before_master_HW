import numpy as np


class auto_encoder:
    def __init__(self, input_size=784, hidden_size=64, noise=False, dropout_rate=0.0):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.noise = noise
        self.dropout_rate = dropout_rate
        self.W1 = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, input_size) / np.sqrt(hidden_size)
        self.b2 = np.zeros((1, input_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1 - x)

    def add_noise(self, x, noise_level=0.2):
        return x + noise_level * np.random.normal(0, 1, x.shape)

    def apply_dropout(self, x):
        mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
        return x * mask

    def forward(self, x):

        if self.noise:
            x = self.add_noise(x)

        if self.dropout_rate > 0:
            x = self.apply_dropout(x)

        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h = self.sigmoid(self.z1)
        self.z2 = self.h @ self.W2 + self.b2
        self.y = self.sigmoid(self.z2)
        return self.y

    def compute_loss(self, y_true):
        return 0.5 * np.mean((y_true - self.y) ** 2)

    def backward(self):
        dy = self.y - self.x
        dz2 = dy * self.sigmoid_deriv(self.y)
        dW2 = self.h.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        dh = dz2 @ self.W2.T
        dz1 = dh * self.sigmoid_deriv(self.h)
        dW1 = self.x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2

    def update(self, dW1, db1, dW2, db2, lr=0.01):
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
