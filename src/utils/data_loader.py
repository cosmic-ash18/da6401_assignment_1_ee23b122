"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
# keras.datasets contains both the mnist datasets
from keras.datasets import mnist, fashion_mnist
from sklearn.model_selection import train_test_split


def one_hot_encode(labels, num_classes=10):
    # If already one-hot, just return
    if labels.ndim == 2 and labels.shape[1] == num_classes:
        return labels
    # convert classes to one-hot encode
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot

# function loads and preprocesses the dataset
# val_split is the fraction of data used for validation
def load_dataset(dataset_name, val_split=0.1):

    # Load and preprocess dataset.

    # The dataset name would be mnist or fashion_mnist

    # such functions usually return X,y_train, val and test
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError("Invalid dataset")

    # image flattening - standard procedure to convert to easy numbers
    X_train = X_train.reshape(-1, 784) / 255.0
    X_test = X_test.reshape(-1, 784) / 255.0

    # the train and test and just the one_hot_encoded versions 
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # Validation split, using the function from sklearn
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_split, random_state=42, shuffle=True
    )

    return X_train, y_train, X_val, y_val, X_test, y_test