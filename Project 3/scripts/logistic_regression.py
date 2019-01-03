#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import numpy as np
from keras.utils import np_utils
from tqdm import tqdm
import confusion_matrix


# Initializing the weights 'normally'
def init_weights(shape):
    return np.random.randn(shape, 10)


# Calculate Softmax
def softmax(z):
    exp = np.exp(z)
    yhat = exp / np.sum(exp, axis=0)
    return yhat.T


# Perform Gradient Descent
def perform_training(features, y, w, learning_rate, num_of_epochs):
    for _ in tqdm(range(num_of_epochs), total=num_of_epochs):
        z = np.dot(w.T, features.T)

        yhat = softmax(z)
        gradient = np.dot(features.T, yhat - y)
        gradient /= len(features.T)
        gradient *= learning_rate
        w -= gradient

    return w


def perform_testing(features, y, w):
    z = np.dot(w.T, features.T)
    yhat = softmax(z)

    num_of_wrong_predictions = 0
    num_of_right_predictions = 0

    for i, j in zip(y, yhat):
        if np.argmax(i) == np.argmax(j):
            num_of_right_predictions = num_of_right_predictions + 1
        else:
            num_of_wrong_predictions = num_of_wrong_predictions + 1

    accuracy = num_of_right_predictions / (num_of_right_predictions + num_of_wrong_predictions)

    return np.argmax(yhat, axis=1), accuracy


def start(training_data, training_label, mnist_test_data, mnist_test_label, usps_test_data, usps_test_label,
          combined_test_data, combined_test_label, num_of_epochs=500, learning_rate=0.01):
    print('\nPerforming Logistic Regression . . .')
    w = init_weights(training_data.shape[1])

    # Perform Training
    w = perform_training(training_data, np_utils.to_categorical(np.array(training_label), 10), w,
                         learning_rate, num_of_epochs)

    # Test on MNIST dataset
    yhat_mnist, accuracy_mnist = perform_testing(mnist_test_data, np_utils.to_categorical(np.array(mnist_test_label), 10), w)
    confusion_mnist = confusion_matrix.calculate(mnist_test_label, yhat_mnist)

    # Test on USPS dataset
    yhat_usps, accuracy_usps = perform_testing(usps_test_data, np_utils.to_categorical(np.array(usps_test_label), 10), w)
    confusion_usps = confusion_matrix.calculate(usps_test_label, yhat_usps)

    # Test on combined datasets
    yhat_combined, accuracy_combined = perform_testing(
        combined_test_data, np_utils.to_categorical(np.array(combined_test_label), 10), w)
    confusion_combined = confusion_matrix.calculate(combined_test_label, yhat_combined)

    return yhat_mnist, accuracy_mnist, confusion_mnist,\
           yhat_usps, accuracy_usps, confusion_usps,\
           yhat_combined, accuracy_combined, confusion_combined
