import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse
import json


def load_data(data_file_name, n_features, n_samples):
    with open(data_file_name) as f:
        data_file = csv.reader(f)
        data = numpy.empty((n_samples, n_features))
        for i, d in enumerate(data_file):
            data[i] = numpy.asarray(d[:], dtype=float)
        f.close()

        return data


def model(data_path, train_num, test_num):
    # train_data_f = args.data_path + "a_a_list_train_feature.txt"
    train_data_f = data_path + "train_feature.txt"
    train_data = load_data(train_data_f, 128 + 3, train_num)
    train_features = train_data.astype(numpy.float32)[:, 3:-1]
    train_target = train_data.astype(numpy.float32)[:, 2]

    # print(train_target[1])
    learner = linear_model.LogisticRegression(max_iter=1000)
    learner.fit(train_features, train_target)
    train_features = None
    train_target = None

    print("training finish!")
    # test_data_f = args.data_path + "a_a_list_test_feature.txt"
    test_data_f = data_path + "test_feature.txt"
    test_data = load_data(test_data_f, 128 + 3, test_num)
    test_id = test_data.astype(numpy.int32)[:, 0:2]
    test_features = test_data.astype(numpy.float32)[:, 3:-1]
    test_target = test_data.astype(numpy.float32)[:, 2]
    test_predict = learner.predict(test_features)
    # print(len(test_predict))
    # open(args.data_path + '/' + 'a_t_list_test_score.json', 'w+').write(json.dumps(list(test_predict), indent=4))
    # exit()
    test_features = None

    print("test prediction finish!")

    output_f = open(data_path + "link_prediction.txt", "w")
    for i in range(len(test_predict)):
        output_f.write('%d, %d, %lf\n' % (test_id[i][0], test_id[i][1], test_predict[i]));
    output_f.close()

    '''
    unique = []
    for i in range(len(test_predict)):
        if test_predict[i] not in unique:
            unique.append(test_predict[i])

    print("unique: ")
    print(unique)
    '''

    #AUC_score = Metric.roc_auc_score(test_target, test_predict)
    #print("AUC: " + str(AUC_score))

    total_count = 0
    correct_count = 0
    true_p_count = 0
    false_p_count = 0
    false_n_count = 0
    true_0_count = 0
    true_1_count = 0
    true_2_count = 0
    all_0_count = 1
    all_1_count = 1
    all_2_count = 1

    for i in range(len(test_predict)):
        total_count += 1
        if int(test_predict[i]) == int(test_target[i]):
            correct_count += 1
        if int(test_target[i]) == 0:
            all_0_count += 1
        if  int(test_target[i]) == 1:
            all_1_count += 1
        if  int(test_target[i]) == 2:
            all_2_count += 1
        if int(test_predict[i]) == 0 and int(test_target[i]) == 0:
            true_0_count += 1
        if int(test_predict[i]) == 1 and int(test_target[i]) == 1:
            true_1_count += 1
        if int(test_predict[i]) == 2 and int(test_target[i]) == 2:
            true_2_count += 1
        if int(test_predict[i]) == 1 and int(test_target[i] == 1):
            true_p_count += 1
        if int(test_predict[i]) == 1 and int(test_target[i]) == 0:
            false_p_count += 1
        if int(test_predict[i]) == 0 and int(test_target[i]) == 1:
            false_n_count += 1

    print("total count: " + str(total_count))
    print("all_0_count: " + str(all_0_count))
    print("all_1_count: " + str(all_1_count))
    print("all_2_count: " + str(all_2_count))
    #print("added: " + str(all_0_count + all_1_count + all_2_count))
    print("accuracy: " + str(float(correct_count) / total_count))
    print("accuracy 0: " + str(float(true_0_count) / all_0_count))
    print("accuracy 1: " + str(float(true_1_count) / all_1_count))
    print("accuracy 2: " + str(float(true_2_count) / all_2_count))

    precision = float(true_p_count) / (true_p_count + false_p_count)
    print("precision: " + str(precision))
    recall = float(true_p_count) / (true_p_count + false_n_count)
    print("recall: " + str(recall))
    F1 = float(2 * precision * recall) / (precision + recall)
    print("F1: " + str(F1))
