import numpy as np
import nerual_network
import matplotlib.pyplot as plt
import struct
import os

mnist_dir = "陳奕_2025_MLHW1/MNIST_data"


def load_images(file_path):
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
    return images


def load_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def one_hot(y):
    return np.eye(10, dtype=np.float32)[y]


def caculate_accuracy(y, y_):
    y_digit = np.argmax(y, 1)
    y_digit_ = np.argmax(y_, 1)
    return np.mean(y_digit == y_digit_)


# load data
x_train = load_images(os.path.join(mnist_dir, "train-images.idx3-ubyte"))
y_train = load_labels(os.path.join(mnist_dir, "train-labels.idx1-ubyte"))
x_test = load_images(os.path.join(mnist_dir, "t10k-images.idx3-ubyte"))
y_test = load_labels(os.path.join(mnist_dir, "t10k-labels.idx1-ubyte"))


# print(x_train[0])
# print(y_train[0])
# plt.imshow(x_train[0].reshape((28, 28)), cmap="gray")
# plt.title(f"Label: {y_train[0]}")
# plt.show()


x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = one_hot(y_train)
y_test = one_hot(y_test)

epoch = 10
print("this is wide neural network")
wide_network = nerual_network.neural_network(784, [256], 10, "softmax")
wide_loss_history = []

for i in range(epoch):
    wide_network.feed({"x": x_train, "y": y_train})
    wide_network.forward()
    loss = wide_network.computeLoss()
    wide_loss_history.append(loss)
    wide_network.backward()
    wide_network.update(learning_rate=0.01)

    y_test_pred = (
        wide_network.feed({"x": x_test, "y": y_test}) or wide_network.forward()
    )
    accuracy = caculate_accuracy(wide_network.y, y_test)
    print(f"Epoch {i+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

print("this is deep neural network")
deep_network = nerual_network.neural_network(784, [204, 202], 10, "softmax")
deep_loss_history = []


for i in range(epoch):
    deep_network.feed({"x": x_train, "y": y_train})
    deep_network.forward()
    loss = deep_network.computeLoss()
    deep_loss_history.append(loss)
    deep_network.backward()
    deep_network.update(learning_rate=0.01)

    y_test_pred = (
        deep_network.feed({"x": x_test, "y": y_test}) or deep_network.forward()
    )
    accuracy = caculate_accuracy(deep_network.y, y_test)
    print(f"Epoch {i+1}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, epoch + 1), wide_loss_history, label="Wide Network")
plt.plot(range(1, epoch + 1), deep_loss_history, label="Deep Network")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
