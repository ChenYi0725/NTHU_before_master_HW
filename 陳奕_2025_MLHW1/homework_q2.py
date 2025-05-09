import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from auto_encoder import auto_encoder


def load_images(file_path):
    with open(file_path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)


def visualize_reconstruction(x, y_pred, n=10):
    plt.figure(figsize=(n, 2))
    for i in range(n):
        # origin
        plt.subplot(2, n, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap="gray")
        plt.axis("off")

        # new
        plt.subplot(2, n, n + i + 1)
        plt.imshow(y_pred[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()


def visualize_filters(W1):
    plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(W1[:, i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()


mnist_dir = "陳奕_2025_MLHW1/MNIST_data"
x_train = load_images(os.path.join(mnist_dir, "train-images.idx3-ubyte"))
x_train = x_train.astype(np.float32) / 255.0

auto_encoder_model = auto_encoder(
    input_size=784, hidden_size=64, noise=True, dropout_rate=0.2
)

epoch = 20
batch_size = 256
losses = []

for e in range(epoch):
    idx = np.random.permutation(len(x_train))
    x_train = x_train[idx]

    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i : i + batch_size]
        y_pred = auto_encoder_model.forward(x_batch)
        loss = auto_encoder_model.compute_loss(x_batch)
        dW1, db1, dW2, db2 = auto_encoder_model.backward()
        auto_encoder_model.update(dW1, db1, dW2, db2, lr=0.01)

    losses.append(loss)
    print(f"Epoch {e+1}, Loss: {loss:.4f}")

x_sample = x_train[:10]
y_sample = auto_encoder_model.forward(x_sample)
visualize_reconstruction(x_sample, y_sample)
visualize_filters(auto_encoder_model.W1)
