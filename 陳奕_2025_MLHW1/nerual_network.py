import numpy as np


class neural_network:
    def __init__(self, input_size, hidden_sizes, output_size, activation):
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        self.num_layers = len(layer_sizes) - 1
        self.activation = activation  # softmax, sigmoid, linear

        # w -> weight, b -> bias
        self.w = []
        self.b = []
        for i in range(self.num_layers):
            self.w.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1])
                / np.sqrt(layer_sizes[i])
            )
            self.b.append(np.zeros((1, layer_sizes[i + 1])))

        self.placeholder = {"x": None, "y": None}
        self.z = []  # 加權輸入
        self.a = []  # 激活輸出

    def feed(self, feed_dict):
        for key in feed_dict:
            self.placeholder[key] = feed_dict[key].copy()

    def forward(self):
        self.z = []
        self.a = [self.placeholder["x"]]

        for i in range(self.num_layers):
            z_i = self.a[i] @ self.w[i] + self.b[i]
            self.z.append(z_i)
            if i == self.num_layers - 1:  # 最後一層
                if self.activation == "softmax":
                    a_i = np.exp(z_i - np.max(z_i, axis=1, keepdims=True))
                    a_i /= np.sum(a_i, axis=1, keepdims=True)
                elif self.activation == "sigmoid":
                    a_i = 1 / (1 + np.exp(-z_i))
                else:  # linear
                    a_i = z_i
            else:  # 隱藏層：ReLU
                a_i = np.maximum(z_i, 0)
            self.a.append(a_i)

        self.y = self.a[-1]
        return self.y

    def computeLoss(self):
        y_true = self.placeholder["y"]
        y_pred = self.y
        if self.activation == "softmax":
            loss = -y_true * np.log(y_pred + 1e-8)
            return np.mean(np.sum(loss, axis=1))
        elif self.activation == "sigmoid":
            loss = -y_true * np.log(y_pred + 1e-8) - (1 - y_true) * np.log(
                1 - y_pred + 1e-8
            )
            return np.mean(loss)
        else:  # linear/MSE
            return 0.5 * np.mean((y_true - y_pred) ** 2)

    def backward(self):
        grads_w = [None] * self.num_layers
        grads_b = [None] * self.num_layers
        y_true = self.placeholder["y"]
        y_pred = self.y
        n = y_true.shape[0]

     
        delta = (y_pred - y_true) / n

    
        for i in reversed(range(self.num_layers)):
            grads_w[i] = self.a[i].T @ delta
            grads_b[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                da = delta @ self.w[i].T
                dz = da * (self.z[i - 1] > 0)  # ReLU 導數
                delta = dz

        self.grads_w = grads_w
        self.grads_b = grads_b

    def update(self, learning_rate=1e-3):
        for i in range(self.num_layers):
            self.w[i] -= learning_rate * self.grads_w[i]
            self.b[i] -= learning_rate * self.grads_b[i]
