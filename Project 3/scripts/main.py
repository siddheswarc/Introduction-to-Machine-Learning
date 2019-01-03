#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import os
import data_preprocessing
import logistic_regression
import neural_network
import svm
import random_forest
import confusion_matrix
import ensemble

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable the warning, doesn't enable AVX/FMA


if __name__ == '__main__':

    # Process dataset
    training_data_features, training_data_labels, validation_data_features, validation_data_labels,\
        mnist_test_data_features, mnist_test_data_labels, usps_test_data_features, usps_test_data_labels,\
        combined_test_data_features, combined_test_data_labels = data_preprocessing.start()

    # Logistic Regression
    yhat_logistic_mnist, accuracy_mnist, confusion_mnist, yhat_logistic_usps, accuracy_usps, confusion_usps,\
    yhat_logistic_combined, accuracy_combined, confusion_combined =\
        logistic_regression.start(training_data_features, training_data_labels,
                                  mnist_test_data_features, mnist_test_data_labels,
                                  usps_test_data_features, usps_test_data_labels,
                                  combined_test_data_features, combined_test_data_labels,
                                  num_of_epochs=500, learning_rate=0.01)

    print('\n\n----------Logistic Regression----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot('Logistic Regression on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot('Logistic Regression on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot('Logistic Regression on combined datasets',
                          confusion_combined, confusion_combined.shape[0])

    # Neural Network
    yhat_nn_mnist, accuracy_mnist, confusion_mnist, yhat_nn_usps, accuracy_usps, confusion_usps, \
    yhat_nn_combined, accuracy_combined, confusion_combined =\
        neural_network.start(training_data_features, training_data_labels,
                             mnist_test_data_features, mnist_test_data_labels,
                             usps_test_data_features, usps_test_data_labels,
                             combined_test_data_features, combined_test_data_labels, num_of_epochs=15)

    print('\n\n----------Neural Network----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot('Neural Network on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot('Neural Network on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot('Neural Network on combined datasets',
                          confusion_combined, confusion_combined.shape[0])

    # SVM
    yhat_svm_mnist, accuracy_mnist, confusion_mnist, yhat_svm_usps, accuracy_usps, confusion_usps, \
    yhat_svm_combined, accuracy_combined, confusion_combined =\
        svm.start(training_data_features, training_data_labels,
                  mnist_test_data_features, mnist_test_data_labels,
                  usps_test_data_features, usps_test_data_labels,
                  combined_test_data_features, combined_test_data_labels,
                  C=1.0, kernel='rbf', gamma='scale')

    print('\n\n----------SVM----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot('SVM on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot('SVM on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot('SVM on combined datasets',
                          confusion_combined, confusion_combined.shape[0])

    # Random Forest
    yhat_rf_mnist, accuracy_mnist, confusion_mnist, yhat_rf_usps, accuracy_usps, confusion_usps, \
    yhat_rf_combined, accuracy_combined, confusion_combined =\
        random_forest.start(training_data_features, training_data_labels,
                            mnist_test_data_features, mnist_test_data_labels,
                            usps_test_data_features, usps_test_data_labels,
                            combined_test_data_features, combined_test_data_labels,
                            n_estimators=100, max_depth=None, random_state=None)

    print('\n\n----------Random Forest----------')
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_mnist))
    confusion_matrix.plot('Random Forest on MNIST dataset',
                          confusion_mnist, confusion_mnist.shape[0])

    print('\nOn USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_usps))
    confusion_matrix.plot('Random Forest on USPS dataset',
                          confusion_usps, confusion_usps.shape[0])

    print('\nOn Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_combined))
    confusion_matrix.plot('Random Forest on combined datasets',
                          confusion_combined, confusion_combined.shape[0])

    # Ensemble of Classifiers
    print('\n\n----------Ensemble of Classifiers----------')

    # On MNIST data
    accuracy_ensemble_mnist, confusion_ensemble_mnist = ensemble.start(
        mnist_test_data_labels, yhat_logistic_mnist, yhat_nn_mnist, yhat_svm_mnist, yhat_rf_mnist)
    print('On MNIST data')
    print('Testing Accuracy: ' + str(round((accuracy_ensemble_mnist * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_ensemble_mnist))
    confusion_matrix.plot('Ensemble on MNIST dataset',
                          confusion_ensemble_mnist, confusion_ensemble_mnist.shape[0])

    # On USPS data
    accuracy_ensemble_usps, confusion_ensemble_usps = ensemble.start(
        usps_test_data_labels, yhat_logistic_usps, yhat_nn_usps, yhat_svm_usps, yhat_rf_usps)
    print('On USPS data')
    print('Testing Accuracy: ' + str(round((accuracy_ensemble_usps * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_ensemble_usps))
    confusion_matrix.plot('Ensemble on USPS dataset',
                          confusion_ensemble_usps, confusion_ensemble_usps.shape[0])

    # On combined data
    accuracy_ensemble_combined, confusion_ensemble_combined = ensemble.start(
        combined_test_data_labels, yhat_logistic_combined, yhat_nn_combined, yhat_svm_combined, yhat_rf_combined)
    print('On Combined data')
    print('Testing Accuracy: ' + str(round((accuracy_ensemble_combined * 100), 2)) + '%')
    print('Confusion Matrix: \n' + str(confusion_ensemble_combined))
    confusion_matrix.plot('Ensemble on Combined datasets',
                          confusion_ensemble_combined, confusion_ensemble_combined.shape[0])
