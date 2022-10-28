# -*- coding: utf-8 -*-
import numpy as np
import sys
import data_utils
import implementations
import csv

train_data, train_label = data_utils.load_data('Data/train.csv')

print('Start Training...')
best_degrees = [9,8,9]
add_bias = [False, True, True]
group0_x, group0_labels, group1_x, group1_labels, group2_x, group2_labels = \
                          data_utils.process_data(train_data, train_label, clean = True, sigma_n = 3.0)
group0_x, group1_x, group2_x = \
        data_utils.group_poly(group0_x, group1_x, group2_x, degree_list = best_degrees, all_poly = True)

group0_x, group1_x, group2_x = data_utils.add_bias_group(group0_x, group1_x, group2_x, add_bias)
group_0_best_lambda = 1e-10
group_0_model = implementations.ridge_regression

group_0_w, _ = group_0_model(group0_labels, group0_x, lambda_ = group_0_best_lambda)

group_1_best_lambda = 1e-10
group_1_model = implementations.ridge_regression
group_1_w, _ = group_1_model(group1_labels, group1_x, lambda_ = group_1_best_lambda)

group_2_best_lambda = 1e-10
group_2_model = implementations.ridge_regression
group_2_w, _ = group_2_model(group2_labels, group2_x, lambda_ = group_2_best_lambda)

print('Start Testing...')
test_data, _, test_id = data_utils.load_data('Data/test.csv', return_id = True)
group0_x_test, _, group1_x_test, _, group2_x_test, _ = \
                          data_utils.process_data(test_data, np.ones(len(test_data)), train = False, clean = True, sigma_n = 3.0)
group0_x_test, group1_x_test, group2_x_test = \
        data_utils.group_poly(group0_x_test, group1_x_test, group2_x_test, degree_list = best_degrees, all_poly = True)
group0_x_test, group1_x_test, group2_x_test = data_utils.add_bias_group(group0_x_test, group1_x_test, group2_x_test, add_bias)
group0_output = implementations.compute_statistics_all(np.ones(len(group0_x_test)), group0_x_test, group_0_w, func_type = 'linear', return_result = True)
group1_output = implementations.compute_statistics_all(np.ones(len(group1_x_test)), group1_x_test, group_1_w, func_type = 'linear', return_result = True)
group2_output = implementations.compute_statistics_all(np.ones(len(group2_x_test)), group2_x_test, group_2_w, func_type = 'linear', return_result = True)

final_output = np.zeros((test_data.shape[0]))
final_output[np.where(test_data[:, 22] == 0)] = group0_output
final_output[np.where(test_data[:, 22] == 1)] = group1_output
final_output[np.where(test_data[:, 22] >= 2)] = group2_output

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
create_csv_submission(test_id, final_output, 'submission.csv')