#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import confusion_matrix


def start(training_data, training_label, mnist_test_data, mnist_test_label, usps_test_data, usps_test_label,
          combined_test_data, combined_test_label, n_estimators=100, max_depth=None, random_state=None):
    print('\nPerforming Random Forest . . .')

    # Classify as Random Forest Model
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Perform training
    classifier.fit(training_data, training_label)

    # Test on MNIST dataset
    yhat_mnist = classifier.predict(mnist_test_data)
    accuracy_mnist = metrics.accuracy_score(mnist_test_label, yhat_mnist)
    confusion_mnist = confusion_matrix.calculate(mnist_test_label, yhat_mnist)

    # Test on USPS dataset
    yhat_usps = classifier.predict(usps_test_data)
    accuracy_usps = metrics.accuracy_score(usps_test_label, yhat_usps)
    confusion_usps = confusion_matrix.calculate(usps_test_label, yhat_usps)

    # Test on combined datasets
    yhat_combined = classifier.predict(combined_test_data)
    accuracy_combined = metrics.accuracy_score(combined_test_label, yhat_combined)
    confusion_combined = confusion_matrix.calculate(combined_test_label, yhat_combined)

    return yhat_mnist, accuracy_mnist, confusion_mnist,\
           yhat_usps, accuracy_usps, confusion_usps,\
           yhat_combined, accuracy_combined, confusion_combined
