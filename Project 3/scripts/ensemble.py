#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


from sklearn import metrics
from statistics import mode
import confusion_matrix


def start(y, yhat_logistic, yhat_nn, yhat_svm, yhat_rf):
    yhat = []

    for i in range(len(y)):
        try:
            yhat.append(mode([yhat_logistic[i], yhat_nn[i], yhat_svm[i], yhat_rf[i]]))

        except:
            yhat.append(yhat_nn[i])

    accuracy = metrics.accuracy_score(y, yhat)
    confusion = confusion_matrix.calculate(y, yhat)

    return accuracy, confusion
