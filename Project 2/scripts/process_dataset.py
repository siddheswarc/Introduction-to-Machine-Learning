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


# Write dataset to CSV file
def write_to_CSV(dataset, path):
    pd.DataFrame(dataset).to_csv(path, header = False, index = False)


# Generate concatenated features dataset for Human Observed and GSC Features
def generate_concat_dataset(features_dataset, data, start, size):
    dataset = pd.DataFrame()

    for i in range (size):
        finda = features_dataset.iloc[: , start:].loc[features_dataset['img_id'] == data.at[i,'img_id_A']]
        findb = features_dataset.iloc[: , start:].loc[features_dataset['img_id'] == data.at[i,'img_id_B']]

        found = finda.reset_index(drop=True).merge(findb.reset_index(drop=True), left_index=True, right_index=True, suffixes = ['a','b'])
        found['target'] = data.at[i,'target']

        dataset = pd.concat([dataset,found], ignore_index = True)

    return dataset


# Generate subtracted features dataset for Human Observed and GSC Features
def generate_difference_dataset(features_dataset, data, start, size):
    dataset = pd.DataFrame()

    for i in range (size):
        finda = features_dataset.iloc[: , start:].loc[features_dataset['img_id'] == data.at[i,'img_id_A']]
        findb = features_dataset.iloc[: , start:].loc[features_dataset['img_id'] == data.at[i,'img_id_B']]

        finda = finda.reset_index()
        findb = findb.reset_index()

        found = finda.subtract(findb)
        found = found.drop(['index'], axis = 1)

        found['target'] = data.at[i,'target']

        dataset = pd.concat([dataset, found], ignore_index = True)

    return dataset


# Merge two datasets
def merge_dataset(dataset1, dataset2):
    return pd.concat([dataset1,dataset2], ignore_index = True)


# Generate feature dataset
def generate_data(dataset):
    return dataset.iloc[:int(dataset.shape[0]), :dataset.shape[1]-1]


# Generate target dataset
def generate_target(dataset):
    return dataset.iloc[:int(dataset.shape[0]), dataset.shape[1]-1]


def start():
    training_percent = 80

    human_observed_features_dataset = read_from_CSV(r'../datasets/HumanObserved-Features-Data/HumanObserved-Features-Data.csv')
    gsc_features_dataset = read_from_CSV(r'../datasets/GSC-Features-Data/GSC-Features.csv')

    human_observed_same_pairs = read_from_CSV(r'../datasets/HumanObserved-Features-Data/same_pairs.csv')
    human_observed_diffn_pairs = read_from_CSV(r'../datasets/HumanObserved-Features-Data/diffn_pairs.csv')

    gsc_same_pairs = read_from_CSV(r'../datasets/GSC-Features-Data/same_pairs.csv')
    gsc_diffn_pairs = read_from_CSV(r'../datasets/GSC-Features-Data/diffn_pairs.csv')

    human_observed_same_pairs_concat_dataset = generate_concat_dataset(human_observed_features_dataset, human_observed_same_pairs, 2, 750)
    human_observed_diffn_pairs_concat_dataset = generate_concat_dataset(human_observed_features_dataset, human_observed_diffn_pairs, 2, 750)

    human_observed_same_pairs_difference_dataset = generate_difference_dataset(human_observed_features_dataset, human_observed_same_pairs, 2, 750)
    human_observed_diffn_pairs_difference_dataset = generate_difference_dataset(human_observed_features_dataset, human_observed_diffn_pairs, 2, 750)

    gsc_same_pairs_concat_dataset = generate_concat_dataset(gsc_features_dataset, gsc_same_pairs, 1, 1000)
    gsc_diffn_pairs_concat_dataset = generate_concat_dataset(gsc_features_dataset, gsc_diffn_pairs, 1, 1000)

    gsc_same_pairs_difference_dataset = generate_difference_dataset(gsc_features_dataset, gsc_same_pairs, 1, 1000)
    gsc_diffn_pairs_difference_dataset = generate_difference_dataset(gsc_features_dataset, gsc_diffn_pairs, 1, 1000)

    human_observed_concat_dataset = merge_dataset(human_observed_same_pairs_concat_dataset, human_observed_diffn_pairs_concat_dataset)
    human_observed_concat_dataset = human_observed_concat_dataset.sample(frac = 1)

    human_observed_difference_dataset = merge_dataset(human_observed_same_pairs_difference_dataset, human_observed_diffn_pairs_difference_dataset)
    human_observed_difference_dataset = human_observed_difference_dataset.sample(frac = 1)

    gsc_cat_dataset = merge_dataset(gsc_same_pairs_concat_dataset, gsc_diffn_pairs_concat_dataset)
    gsc_cat_dataset = gsc_cat_dataset.sample(frac = 1) # Shuffling the dataset
    gsc_cattest_dataset = gsc_cat_dataset.iloc[:int(math.ceil(gsc_cat_dataset.shape[0]*training_percent*0.01)), : ]

    # Dropping features that give same value for every sample (Features that do not contribute to our model)
    cols = list(gsc_cattest_dataset)
    nunique = gsc_cattest_dataset.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    gsc_concat_dataset = gsc_cat_dataset.drop(cols_to_drop, axis = 1)

    gsc_diff_dataset = merge_dataset(gsc_same_pairs_difference_dataset, gsc_diffn_pairs_difference_dataset)
    gsc_diff_dataset = gsc_diff_dataset.sample(frac = 1) # Shuffling the dataset
    gsc_difftest_dataset = gsc_diff_dataset.iloc[:int(math.ceil(gsc_diff_dataset.shape[0]*training_percent*0.01)), : ]

    # Dropping features that give same value for every sample (Features that do not contribute to our model)
    cols = list(gsc_difftest_dataset)
    nunique = gsc_difftest_dataset.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    gsc_difference_dataset = gsc_diff_dataset.drop(cols_to_drop, axis = 1)

    human_observed_concat_raw_data = generate_data(human_observed_concat_dataset)
    human_observed_concat_raw_target = generate_target(human_observed_concat_dataset)

    human_observed_difference_raw_data = generate_data(human_observed_difference_dataset)
    human_observed_difference_raw_target = generate_target(human_observed_difference_dataset)

    gsc_concat_raw_data = generate_data(gsc_concat_dataset)
    gsc_concat_raw_target = generate_target(gsc_concat_dataset)

    gsc_difference_raw_data = generate_data(gsc_difference_dataset)
    gsc_difference_raw_target = generate_target(gsc_difference_dataset)

    write_to_CSV(human_observed_concat_raw_data, r'../datasets/human_observed_concat_raw_data.csv')
    write_to_CSV(human_observed_concat_raw_target, r'../datasets/human_observed_concat_raw_target.csv')

    write_to_CSV(human_observed_difference_raw_data, r'../datasets/human_observed_difference_raw_data.csv')
    write_to_CSV(human_observed_difference_raw_target, r'../datasets/human_observed_difference_raw_target.csv')

    write_to_CSV(gsc_concat_raw_data, r'../datasets/gsc_concat_raw_data.csv')
    write_to_CSV(gsc_concat_raw_target, r'../datasets/gsc_concat_raw_target.csv')

    write_to_CSV(gsc_difference_raw_data, r'../datasets/gsc_difference_raw_data.csv')
    write_to_CSV(gsc_difference_raw_target, r'../datasets/gsc_difference_raw_target.csv')
