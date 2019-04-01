#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import os

import models
from confusion_matrix import ConfusionMatrix
from dataset import Data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable the warning, doesn't enable AVX/FMA
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def print_and_plot(model, accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined):
    print('\n\n----------' + model + '----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot(model + ' on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot(model + ' on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot(model + ' on combined dataset',
                          confusion_combined, confusion_combined.shape[0])


def main():
    data = Data()
    logistic_regression = models.LogisticRegression()
    neural_network = models.NeuralNet()
    svm = models.SupportVectorMachine(C=1.0, kernel='rbf', gamma='scale')
    random_forest = models.RandomForest(n_estimators=100, max_depth=None, random_state=None)

    # Process dataset
    training_data_features, training_data_labels, mnist_test_data_features, mnist_test_data_labels, \
    usps_test_data_features, usps_test_data_labels, combined_test_data_features, combined_test_data_labels = \
        data.pre_process()

    # Logistic Regression
    logistic_regression.fit(training_data_features, training_data_labels, learning_rate=0.01, epochs=500)
    accuracy_mnist, confusion_mnist = logistic_regression.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = logistic_regression.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = logistic_regression.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Logistic Regression', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Neural Network
    neural_network.fit(training_data_features, training_data_labels, epochs=10)
    accuracy_mnist, confusion_mnist = neural_network.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = neural_network.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = neural_network.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Neural Network', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Support Vector Machine
    svm.fit(training_data_features, training_data_labels)
    accuracy_mnist, confusion_mnist = svm.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = svm.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = svm.predict(combined_test_data_features, combined_test_data_labels)
    print_and_plot('SVM', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)

    # Random Forest
    random_forest.fit(training_data_features, training_data_labels)
    accuracy_mnist, confusion_mnist = random_forest.predict(mnist_test_data_features, mnist_test_data_labels)
    accuracy_usps, confusion_usps = random_forest.predict(usps_test_data_features, usps_test_data_labels)
    accuracy_combined, confusion_combined = random_forest.predict(
        combined_test_data_features, combined_test_data_labels)
    print_and_plot('Random Forest', accuracy_mnist, accuracy_usps, accuracy_combined,
                   confusion_mnist, confusion_usps, confusion_combined)


if __name__ == '__main__':
    confusion_matrix = ConfusionMatrix()
    main()
