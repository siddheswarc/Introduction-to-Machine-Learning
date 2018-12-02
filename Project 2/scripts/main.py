#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
import math
import process_dataset
import linear_regression
import logistic_regression
import neural_network


if __name__ == '__main__':
    process_dataset.start()
    linear_regression.start()
    logistic_regression.start()
