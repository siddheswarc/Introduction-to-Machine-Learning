#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import pandas as pd
import numpy as np
import math


# Read dataset from CSV file
def read_from_CSV(path):
    csv_file = pd.read_csv(path)
    return csv_file


# Generate training dataset for Human Observed and GSC Concatenated and Subtracted Features
def generate_training_data(data, percent):
    return data.iloc[:int(math.ceil(data.shape[0]*percent*0.01)), :]


# Generate training target for Human Observed and GSC Concatenated and Subtracted Features
def generate_training_target(data, percent):
    return data.iloc[:int(math.ceil(data.shape[0]*percent*0.01)), :]


# Generate testing dataset for Human Observed and GSC Concatenated and Subtracted Features
def generate_testing_data(data, percent):
    return data.iloc[int(math.ceil(data.shape[0]*percent*0.01)):, :]


# Generate testing target for Human Observed and GSC Concatenated and Subtracted Features
def generate_testing_target(data, percent):
    return data.iloc[int(math.ceil(data.shape[0]*percent*0.01)):, :]


# Initializing the weights 'normally'
def init_weights(shape):
    return np.random.randn(shape)


# Calculate Sigmoid
def sigmoid(z):
    return (1 / (1 + np.exp(-z)))


# Perform Gradient Descent
def gradient_descent(features, y, w, learning_rate, epochs):
    for i in range (epochs):
        z = np.dot(w, features.T)
        yhat = sigmoid(z)

        gradient = np.dot(features.T, yhat - y)
        gradient /= len(features.T)
        gradient *= learning_rate
        w -= gradient

    return w


# Calculate Mean Squared Error (MSE)
def calculate_erms(y, yhat):
    sum = 0

    for i in range (len(y)):
        sum = sum + math.pow((y[i] - yhat[i]), 2)

    return math.sqrt(sum / len(y))


# Calculate accuracy
def test_accuracy(features, y, w):
    z = np.dot(w, features.T)
    yhat = sigmoid(z)
    erms = calculate_erms(y, yhat)
    yhat = np.around(yhat)

    return len([ i for i, j in zip(yhat, y) if i == j]) / len(y), erms


def perform_logistic_regression(dataset, target, training_percent, learning_rate, epochs):
    training_data = generate_training_data(dataset, training_percent).values
    training_target = generate_training_target(target, training_percent).values
    testing_data = generate_testing_data(dataset, training_percent).values
    testing_target = generate_testing_target(target, training_percent).values

    y = []

    for i in range (len(training_target)):
        y.append(training_target[i][0])

    y = np.asarray(y)
    w = init_weights(training_data.shape[1])
    w = gradient_descent(training_data, y, w, learning_rate, epochs)

    accuracy, erms = test_accuracy(testing_data, testing_target, w)

    return math.ceil(accuracy * 100), erms


def start():

    learning_rate = 0.01
    epochs = 1000

    training_percent = 80
    test_percent = 20

    human_observed_concat_raw_data = read_from_CSV(r'../datasets/human_observed_concat_raw_data.csv')
    human_observed_concat_raw_target = read_from_CSV(r'../datasets/human_observed_concat_raw_target.csv')

    human_observed_difference_raw_data = read_from_CSV(r'../datasets/human_observed_difference_raw_data.csv')
    human_observed_difference_raw_target = read_from_CSV(r'../datasets/human_observed_difference_raw_target.csv')

    gsc_concat_raw_data = read_from_CSV(r'../datasets/gsc_concat_raw_data.csv')
    gsc_concat_raw_target = read_from_CSV(r'../datasets/gsc_concat_raw_target.csv')

    gsc_difference_raw_data = read_from_CSV(r'../datasets/gsc_difference_raw_data.csv')
    gsc_difference_raw_target = read_from_CSV(r'../datasets/gsc_difference_raw_target.csv')

    # Logistic Regression on Human Observed Concatenated Features
    accuracy, erms = perform_logistic_regression(human_observed_concat_raw_data, human_observed_concat_raw_target, training_percent, learning_rate, epochs)
    print ('-------Logistic Regression on Human Observed Concatenated Features-------')
    print ('Accuracy : ' + str(accuracy) + '%' + '\t Erms : ' + str(erms))
    print ()

    # Logistic Regression on Human Observed Subtracted Features
    accuracy, erms = perform_logistic_regression(human_observed_difference_raw_data, human_observed_difference_raw_target, training_percent, learning_rate, epochs)
    print ('-------Logistic Regression on Human Observed Subtracted Features-------')
    print ('Accuracy : ' + str(accuracy) + '%' + '\t Erms : ' + str(erms))
    print ()

    # Logistic Regression on GSC Concatenated Features
    accuracy, erms = perform_logistic_regression(gsc_concat_raw_data, gsc_concat_raw_target, training_percent, learning_rate, epochs)
    print ('-------Logistic Regression on GSC Concatenated Features-------')
    print ('Accuracy : ' + str(accuracy) + '%' + '\t Erms : ' + str(erms))
    print ()

    # Logistic Regression on GSC Subtracted Features
    accuracy, erms = perform_logistic_regression(gsc_difference_raw_data, gsc_difference_raw_target, training_percent, learning_rate, epochs)
    print ('-------Logistic Regression on GSC Subtracted Features-------')
    print ('Accuracy : ' + str(accuracy) + '%' + '\t Erms : ' + str(erms))
    print ()


if __name__ == '__main__':
    start()
