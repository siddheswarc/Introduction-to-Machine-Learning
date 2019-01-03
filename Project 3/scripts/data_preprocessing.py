#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Siddheswar C
# @Email: innocentdevil.sid007@gmail.com


import pickle
import gzip
from PIL import Image
import os
import numpy as np


def start():
    print('processing dataset . . .')
    f = gzip.open(r'../datasets/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    usps_features = []
    usps_labels = []
    cur_path = r'../datasets/USPSdata/Numerals'

    for j in range(0, 10):
        cur_folder_path = cur_path + r'/' + str(j)
        imgs = os.listdir(cur_folder_path)

        for img in imgs:
            cur_img = cur_folder_path + r'/' + img

            if cur_img[-3:] == 'png':
                img = Image.open(cur_img, 'r')
                img = img.resize((28, 28))
                imgdata = (255 - np.array(img.getdata())) / 255
                usps_features.append(imgdata)
                usps_labels.append(j)

    testing_data = np.concatenate((test_data[0], np.asarray(usps_features)), axis=0)
    testing_label = np.concatenate((test_data[1], np.asarray(usps_labels)), axis=0)

    print('data processing complete !')
    print()

    return training_data[0], training_data[1], validation_data[0], validation_data[1],\
           test_data[0], test_data[1], np.asarray(usps_features), np.asarray(usps_labels), testing_data, testing_label
