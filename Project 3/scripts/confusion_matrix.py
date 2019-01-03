#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


# Calculate Confusion Matrix
def calculate(y, yhat):
    return confusion_matrix(y, yhat)


# Plot Confusion Matrix
def plot(title, confusion, matrix_dimension):
    plt.clf()
    plt.xlabel('Actual Target (y)')
    plt.ylabel('Predicted Target (yhat)')
    plt.grid(False)
    plt.xticks(np.arange(matrix_dimension))
    plt.yticks(np.arange(matrix_dimension))
    plt.title(title)
    plt.imshow(confusion, cmap=plt.cm.jet, interpolation='nearest')

    for i, cas in enumerate(confusion):
        for j, count in enumerate(cas):
            if count > 0:
                xoff = .07 * len(str(count))
                plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')

    plt.savefig('../confusion_matrices/' + str(title))
