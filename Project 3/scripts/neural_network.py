#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import tensorflow as tf
from sklearn import metrics
import numpy as np
import confusion_matrix


def start(training_data, training_label, mnist_test_data, mnist_test_label, usps_test_data, usps_test_label,
          combined_test_data, combined_test_label, num_of_epochs=5):
    print('\nPerforming Neural Networking . . .')
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Perform Training
    model.fit(training_data, training_label, epochs=num_of_epochs)

    # Test on MNIST dataset
    yhat_mnist = np.argmax(model.predict(mnist_test_data), axis=1)
    accuracy_mnist = metrics.accuracy_score(mnist_test_label, yhat_mnist)
    confusion_mnist = confusion_matrix.calculate(mnist_test_label, yhat_mnist)

    # Test on USPS dataset
    yhat_usps = np.argmax(model.predict(usps_test_data), axis=1)
    accuracy_usps = metrics.accuracy_score(usps_test_label, yhat_usps)
    confusion_usps = confusion_matrix.calculate(usps_test_label, yhat_usps)

    # Test on combined datasets
    yhat_combined = np.argmax(model.predict(combined_test_data), axis=1)
    accuracy_combined = metrics.accuracy_score(combined_test_label, yhat_combined)
    confusion_combined = confusion_matrix.calculate(combined_test_label, yhat_combined)

    return yhat_mnist, accuracy_mnist, confusion_mnist,\
           yhat_usps, accuracy_usps, confusion_usps,\
           yhat_combined, accuracy_combined, confusion_combined
