import gzip
import numpy as np


def load_images(path):
    """Read an MNIST image file (idx3-ubyte.gz) into an array of shape (n, 784)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Header: magic number, number of images, rows, cols (4 bytes each, big-endian)
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(n, rows * cols)

def load_labels(path):
    """Read an MNIST label file (idx1-ubyte.gz) into an array of shape (n,)."""
    with gzip.open(path, "rb") as f:
        data = f.read()
    # Header: magic number, number of labels (4 bytes each)
    return np.frombuffer(data, dtype=np.uint8, offset=8)


if __name__ == "__main__":
    X_train = load_images("../data/mnist/train-images-idx3-ubyte.gz") / 255.0
    y_train = load_labels("../data/mnist/train-labels-idx1-ubyte.gz")
    X_test = load_images("../data/mnist/t10k-images-idx3-ubyte.gz") / 255.0
    y_test = load_labels("../data/mnist/t10k-labels-idx1-ubyte.gz")
    print("Train:", X_train.shape, y_train.shape)  # expected (60000, 784) (60000,)
    print("Test: ", X_test.shape, y_test.shape)    # expected (10000, 784) (10000,)
    print("First labels:", y_train[:10])