import numpy as np

np.set_printoptions(threshold=100000)
import csv
import random
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import time

start_time = time.time()

root_path = 'D:\PycharmProjects\sarcopenia_prediction\BiLSTM/'
data_path = root_path+'data_20220603_1304/id_label_bi_20220603_digital_1304.csv'


def over_sampling(x_train, y_train):
    over_sampling_train = []
    train_0_id = []  # 0-records positions
    train_1_id = []  # 1-records positions
    for l, label in enumerate(y_train):
        if label == 0:
            train_0_id.append(l)
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
        else:
            train_1_id.append(l)
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
            over_sampling_train.append(np.append([label], x_train[l], axis=0))
    random.seed(555)
    random.shuffle(over_sampling_train)

    over_sampling_x_train = [item[1:] for item in over_sampling_train]
    over_sampling_y_train = [item[0] for item in over_sampling_train]
    return over_sampling_x_train, over_sampling_y_train


# check label-tf_feature record
x = []
y = []
with open(data_path, newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(spamreader):
        if not row:
            continue
        row = [float(item) for item in row]
        y.append(row[0])
        x.append(row[1:])

x = np.array(x)
y = np.array(y)
overall_dict = {}
total_total_print = []


# split data into 70:10:20
x_train0, x_test, y_train0, y_test = train_test_split(x, y, random_state=99, train_size=(8 / 10), shuffle=True)  # 88

x_train, x_val, y_train, y_val = train_test_split(x_train0, y_train0, random_state=99, train_size=(7 / 8), shuffle=True)
# stp
# reduce some features; split above first, them reduce
x_train = x_train  # np.delete(x, [868, 1082], 1)
x_val = x_val  # np.delete(x, [868, 1082], 1)
x_test = x_test  # np.delete(x, [868, 1082], 1)
# print(np.shape(x_train), np.shape(x_val), np.shape(x_test))  # 1031=(1304, 378); 1105=(912, 369) (131, 369) (261, 369)

y1 = [item for item in y_train if item == 1]
y2 = [item for item in y_val if item == 1]
y3 = [item for item in y_test if item == 1]

# print((np.shape(x_train)[0]/np.shape(x_val)[0]), (np.shape(x_test)[0]/np.shape(x_val)[0]))  # (7,2)
# print(len(y1), len(y2), len(y3), (len(y1)/len(y2)), (len(y3)/len(y2)))  # random_state=88=(177 25 47)=(7.08, 1.88)

over_sampling_x_train, over_sampling_y_train = over_sampling(x_train, y_train)

# model train and validation
for allow_iter in [10]:  # 15,50
    for left_ratio in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:  # 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75
        for lr in [0.001,0.005, 0.01, 0.05]:  # , 0.01, 0.05, 0.1]:  # 0.001,0.005,0.01
            # fill parameter
            right_ratio = 1 - left_ratio
            model = LogisticRegression(random_state=6, n_jobs=-1,
                                       class_weight={0: left_ratio, 1: right_ratio}, max_iter=allow_iter, C=lr)
            # train model on the training set
            model.fit(over_sampling_x_train, over_sampling_y_train)
            # validate the trained model via validation set
            prob_val = model.predict_proba(x_val)
            prob_1_val = [item[1] for item in prob_val]  # prob_test = [value0, value1]
            # convert val prediction to roc score
            fpr, tpr, threshold = metrics.roc_curve(y_val, prob_1_val)
            roc_auc = metrics.auc(fpr, tpr)

            # get training set roc score
            prob_train = model.predict_proba(x_train)
            prob_1_train = [item[1] for item in prob_train]  # prob_test = [value0, value1]
            fpr0, tpr0, threshold0 = metrics.roc_curve(y_train, prob_1_train)
            roc_auc0 = metrics.auc(fpr0, tpr0)

            roc_val = np.round(roc_auc * 100, 2)
            roc_train = np.round(roc_auc0 * 100, 2)

            """allow_iter left_ratio lr """

            print('allow_iter=', allow_iter, 'left_ratio=', left_ratio, 'learning rate', lr, '&', roc_val, '&',
                  roc_train)

            parameters = 'allow_iter=' + str(allow_iter) + ', left_ratio=' + str(left_ratio) + ', learning rate=' + str(lr)
            roc_values = roc_val, roc_train
            if 0 < roc_values[1] - roc_values[0] <= 10:
                overall_dict[parameters] = roc_values

print('\n', '-' * 50, '\n')

parameter_list = []
for top_parameter, roc_values in sorted(overall_dict.items(), key=lambda item: item[1][0], reverse=False):
    # print(top_parameter, '&', roc_values[0], '&', roc_values[1])
    parameter_list.append([top_parameter, roc_values[0], roc_values[1]])

parameter_list = parameter_list[-10:]

roc_test_list = []
sensitivity_list = []
specificity_list = []
for best_parameter, roc_val, roc_train in parameter_list:
    new_parameter0 = [item.split('=')[1] for item in best_parameter.split(',')]
    new_parameter = [int(new_parameter0[0]), float(new_parameter0[1]), float(new_parameter0[2])]
    print('Best parameters:', best_parameter)

    # testing
    allow_iter = new_parameter[0]
    left_ratio = new_parameter[1]
    lr = new_parameter[2]
    right_ratio = 1 - left_ratio
    model = LogisticRegression(random_state=6, n_jobs=-1,
                               class_weight={0: left_ratio, 1: right_ratio}, max_iter=allow_iter, C=lr)
    model.fit(over_sampling_x_train, over_sampling_y_train)
    # get testing set score
    prob_test = model.predict_proba(x_test)
    prob_1_test = [item[1] for item in prob_test]  # prob_test = [value0, value1]
    # convert prediction to roc score
    fpr1, tpr1, threshold1 = metrics.roc_curve(y_test, prob_1_test)
    roc_test = metrics.auc(fpr1, tpr1)
    roc_test = np.round(roc_test * 100, 2)
    roc_test_list.append(roc_test)

    y_pred = [0 if value < 0.5 else 1 for value in prob_1_test]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    sensitivity_list.append(sensitivity)
    specificity_list.append(specificity)
    print('test: ', roc_test, 'val: ', roc_val, 'train: ', roc_train)

print('\n', '-' * 50, '\n')
print('sensitivity:', np.mean(sensitivity_list))
print('specificity:', np.mean(specificity_list))
roc_val_list = [item[1] for item in parameter_list]
roc_train_list = [item[2] for item in parameter_list]
method = 'lg'
print('test_list_'+method+'=', roc_test_list, '#', np.mean(roc_test_list))
print('val_list_'+method+'=', roc_val_list, '#', np.mean(roc_val_list))
print('train_list_'+method+'=', roc_train_list, '#', np.mean(roc_train_list))
