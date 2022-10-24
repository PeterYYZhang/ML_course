# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sys
import sys

eps = sys.float_info.epsilon  # a small number for numerical stability


def load_data(path):
    label = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    all_data = np.genfromtxt(path, delimiter=",", skip_header=1)

    data = all_data[:, 2:]

    labels = np.ones(len(label))
    labels[np.where(label == 'b')] = -1

    return data, labels


def group_data_jetnum(x, labels):
    feature_idx_dict = {0: [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21],
                        1: [0, 1, 2, 3, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 29],
                        2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25,
                            26, 27, 28, 29]}

    jetnum = 22

    group0 = x[np.where(x[:, jetnum] == 0)]
    group0_x = group0[:, feature_idx_dict[0]]
    group0_labels = labels[np.where(x[:, jetnum] == 0)]

    group1 = x[np.where(x[:, jetnum] == 1)]
    group1_x = group1[:, feature_idx_dict[1]]
    group1_labels = labels[np.where(x[:, jetnum] == 1)]

    group2 = x[np.where(x[:, jetnum] >= 2)]
    group2_x = group2[:, feature_idx_dict[2]]
    group2_labels = labels[np.where(x[:, jetnum] >= 2)]

    return group0_x, group0_labels, group1_x, group1_labels, group2_x, group2_labels


def handle_outlier(x, n=3, how="sigma"):
    '''
    Input:
        x: input data

        n:
        n = 3 for n-sigma
        n usually <= 0.5 when using IQR

        how: "sigma" / "IQR"

    Output:
        sigma:
    '''
    D = x.shape[1]
    method = how
    if method == "sigma":
        for dim in range(D):
            cur_dim_x = x[:, dim]

            cur_mean = np.mean(cur_dim_x)
            cur_std = np.std(cur_dim_x)
            sigma_3 = cur_std * float(n)

            mean_3sigma_up = cur_mean + sigma_3
            mean_3sigma_below = cur_mean - sigma_3

            data_inside_3sigma = [v for v in cur_dim_x if (v >= mean_3sigma_below) and (v <= mean_3sigma_up)]

            no_outlier = len(data_inside_3sigma) == 0
            if not no_outlier:
                cur_dim_x[np.where(cur_dim_x > mean_3sigma_up)] = np.max(data_inside_3sigma)
                cur_dim_x[np.where(cur_dim_x < mean_3sigma_below)] = np.min(data_inside_3sigma)

            x[:, dim] = cur_dim_x
    elif method == "IQR":
        for dim in range(D):
            cur_dim_x = x[:, dim]

            quantile1 = np.quantile(cur_dim_x, 0.25, axis=0)
            quantile2 = np.quantile(cur_dim_x, 0.75, axis=0)
            iqr = quantile2 - quantile1  # compute the Inter-quartile Range (IQR)
            lower_bound = quantile1 - n * iqr
            upper_bound = quantile2 + n * iqr

            mean = np.mean(cur_dim_x)

            data_within_range = [v for v in cur_dim_x if (v >= lower_bound) and (v <= upper_bound)]
            no_outlier = len(data_within_range)
            if not no_outlier:
                cur_dim_x[np.where(cur_dim_x > upper_bound)] = np.max(data_within_range)
                cur_dim_x[np.where(cur_dim_x < lower_bound)] = np.min(data_within_range)

            x[:, dim] = cur_dim_x
    return x


def build_log(x):
    '''log(x) as feature'''
    N = x.shape[0]
    D = x.shape[1]
    x_log = np.zeros((N, 2 * D))
    x_log[:, :D] = x
    x_log[:, D:] = np.log(x)

    return x_log


def build_poly(x, degree):
    '''Augment x with poly basis function'''
    N = x.shape[0]
    D = x.shape[1]
    x_poly = np.zeros((N, D * (degree)))
    for d in range(1, degree + 1):
        x_poly[:, (d - 1) * D:(d) * D] = np.power(x, d)
    return x_poly


def group_poly(group0_x, group1_x, group2_x, degree_list, all_poly=True):
    '''Due to the different dimensionaility of features, perform the '''
    if (all_poly):
        group0_x = build_poly(group0_x, degree_list[0])
        group1_x = build_poly(group1_x, degree_list[1])
        group2_x = build_poly(group2_x, degree_list[2])

    elif (all_poly == False):
        group0_x_poly = build_poly(group0_x[:, :10], degree_list[0])
        group0_x = np.concatenate([group0_x_poly, group0_x[:, 10:]], axis=1)

        group1_x_poly = build_poly(group1_x[:, :10], degree_list[1])
        group1_x = np.concatenate([group1_x_poly, group1_x[:, 10:]], axis=1)

        group2_x_poly = build_poly(group2_x[:, :13], degree_list[2])
        group2_x = np.concatenate([group2_x_poly, group2_x[:, 13:]], axis=1)

    return group0_x, group1_x, group2_x


def standardize(x):
    mean_x = np.mean(x, axis=0, keepdims=True)
    x = x - mean_x
    std_x = np.std(x, axis=0, keepdims=True)
    x = x / (std_x + eps)
    return x


def normalize(x):
    D = x.shape[1]
    for d in range(D):
        if (np.all(x[:, d] == 1)):
            '''When build poly there can be all 1 column.'''
            pass
        else:
            x[:, d] = (x[:, d] - np.min(x[:, d])) / (np.max(x[:, d]) - np.min(x[:, d]))
    return x


def process_data(x, labels, train=True, clean=True):
    if (train):
        broken_feature = x[:, 0]
        broken_feature_no_nan = broken_feature[np.where(broken_feature != -999.0)]
        fill_na_value = np.median(broken_feature_no_nan)
        broken_feature[np.where(broken_feature == -999.0)] = fill_na_value
        x[:, 0] = broken_feature

    group0_x, group0_labels, group1_x, group1_labels, group2_x, group2_labels = group_data_jetnum(x, labels)

    if (clean):
        group0_x = handle_outlier(group0_x)
        group1_x = handle_outlier(group1_x)
        group2_x = handle_outlier(group2_x)

    return group0_x, group0_labels, group1_x, group1_labels, group2_x, group2_labels


def k_fold(data_num, fold, seed=114514):
    '''Return a array containing {fold} elements of indices'''
    np.random.seed(seed)
    test_num = data_num // fold
    idx_list = np.random.permutation(data_num)
    fold_part_idx = np.array([idx_list[x * test_num: (x + 1) * test_num] for x in range(fold)])
    return fold_part_idx



