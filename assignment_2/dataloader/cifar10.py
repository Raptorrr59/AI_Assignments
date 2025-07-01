import pickle
import numpy as np
import os

def load_batch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
        images = data[b'data']
        labels = data[b'labels']
        images = images.reshape(-1, 3, 32, 32).astype('float32') / 255.0  # Normalize
        return images, np.array(labels)

def load_cifar10(data_dir):
    x_train, y_train = [], []

    for i in range(1, 6):
        imgs, labels = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        x_train.append(imgs)
        y_train.append(labels)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = load_batch(os.path.join(data_dir, 'test_batch'))

    return x_train, y_train, x_test, y_test

def one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]
