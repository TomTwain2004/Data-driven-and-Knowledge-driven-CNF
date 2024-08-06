"""this collection of code include every general tools we use daily so that new definition can save a lot of time."""
import os
import csv
import sys
# csv.field_size_limit(sys.maxsize)  # this one give warning OverflowError: Python int too large to convert to C long
# use below one
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime


def get_files(path):
    files = os.listdir(path)
    return files


def read_csv(file):
    rows = []
    with open(file, newline='', encoding='utf-8') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            rows.append(row)
    return rows


def remove_space(text):
    # print('a'+text+'b')
    while text[0] == ' ':
        text = text[1:]
    while text[-1] == ' ' or text[-1] == '\n':
        text = text[:-1]
    return text


def remove_head_tail_space(text):
    # print('a'+text+'b')
    while text[0] == ' ':
        text = text[1:]
    while text[-1] == ' ' or text[-1] == '\n':
        text = text[:-1]
    return text


def load_keys(my_key):
    keys = []
    with open(my_key, "r") as f:
        for line in f:
            # print([line])
            if not line.replace(' ', '') or not line.replace('\n', ''):
                continue
            k = remove_head_tail_space(line)
            keys.append(k.lower())
    return keys


def f1(a, b):
    return a * b * 2 / (a + b)


def read_text(path):
    output = ''
    f = open(path, encoding='utf-8')
    for line in f:
        output += line
    f.close()
    return output


def clean_punc_number(sentences):
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    return clean_sentences


def remove_stopwords(input):
    stopwords = []
    my_file = r'D:\PycharmProjects\sarcopenia_predictipn\UGIR_stopwords.txt'
    with open(my_file, "r") as f:
        for line in f:
            if line:
                stopwords.append(line.replace('\n', ''))
    # 319, zhuyaoshi daici, guanci, lianci, jieci
    input = input.split()
    sen_new = " ".join([i for i in input if i not in stopwords])
    return sen_new


def csv_writer(path, write_type):
    # if not add newline='', Windows system will add block line follow each line
    w = csv.writer(open(path,  write_type, newline='', encoding='utf-8'))
    return w


def remove_punc(input_text):
    sentences = [input_text]
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")
    return clean_sentences[0]


def quick_plot(y_list, x_list=None, show=True):
    if not x_list:
        x_list = [int(item+1) for item in list(range(len(y_list)))]
    plt.plot(x_list, y_list)
    if show:
        plt.show()
    else:
        plt.savefig()

# above before 0403, below after 0403


def make_dir(input_path):
    if not os.path.exists(input_path):
        os.makedirs(input_path)


# above before 0412, below after 0412

def cut_off_percent(input_list, percent=25):
    cut_off = np.percentile(input_list, [percent])[0]
    return cut_off


def clean_head_tail_block(input_str):
    for i in range(99):
        if input_str[0] == ' ':
            input_str = input_str[1:]
        else:
            break
    for j in range(99):
        if input_str[-1] == ' ':
            input_str = input_str[:-1]
        else:
            break

    return input_str


def compare_two_list(list_1, list_2):

    unique_1 = []
    unique_2 = []

    for item in list_1:
        if item not in list_2:
            unique_1.append(item)

    for item in list_2:
        if item not in list_1:
            unique_2.append(item)

    print('unique in list 1:')
    print(unique_1)
    print('-'*20, '\nunique in list 2:')
    print(unique_2)


# ------ 20220706
def merge_dict(x, y):
    # print(x)
    # print(y)
    # keys = x.keys() + y.keys()
    for k, v in x.items():
        if k in y.keys():
            y[k] = y[k] + v
        else:
            y[k] = v
    # print(y)
    return y


def merge_merge_dict(x, y):
    for km, vm in x.items():
        if km in y.keys():
            merge_dict(vm, y[km])
        else:
            y[km] = vm
    return y


# 20220728

def random_numbers(want_number, total_number):
    if want_number > total_number:
        print('error')
    # to generate 'want_number' random numbers out of 'total number'
    generated = []
    total_number_list = list(range(total_number))
    while len(generated) < want_number:
        new = random.choice(total_number_list)
        if new in generated:
            a = 1
        else:
            generated.append(new)
    return generated


# 20220810
import nltk
# nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()


def stem(sentence):
    # sentence = "Programmers program with programming languages"
    words = word_tokenize(sentence)  # if not token, it will just stem the last word
    output = []
    for w in words:
        # print(w, " : ", ps.stem(w))
        output.append(ps.stem(w))
    return ' '.join(output)


def inverse_the_dict(input_dict):
    output_dict = {}
    for k, v in input_dict.items():
        if v in output_dict.keys():
            output_dict[v] += [k]
        else:
            output_dict[v] = [k]
    return output_dict

# plot elbow curve
"""import matplotlib.pyplot as plt
# Plot the elbow
plt.figure(figsize=(8, 5))
plt.cla()
plt.plot(x1, x2)#, 'bx-')
from kneed import KneeLocator
kn = KneeLocator(x1, x2, S=2.2614864350584, curve='convex', direction='decreasing')  #6~14=95, 15~30=121  (11-17, 33, 37, 58), S=3
print('elbow', kn.knee)  # k value
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
# plt.ylim(-0.05,3.65)
# plt.yticks([item/5 for item in list(range(0,18,1))])
x_tick = list(range(0, 360, 10))
plt.xticks(x_tick, rotation=90)
plt.xlabel('k')  # 62 # best k is c's value, if c starts at 301, k will be 362
plt.ylabel('Importance')
plt.title('The Elbow Method showing the optimal k=42 of f-score')  # k=42(20220126)

plt.tight_layout()
# plt.show()
plt.savefig('/home/resadmin/haoran/BiLSTM/data_20220120/elbow_f_score_20220126.png', dpi=300)
from kneed import KneeLocator, DataGenerator
kneedle = KneeLocator(x, y, S=1.0, curve='concave', direction='increasing')"""


# bert mutual attn tips
"""(1) downgrade to bert-tensorflow==1.0.1
(2) tf.gfile.* is replaced by tf.io.gfile.*
(3) tf.io.gfile.makedirs(data_dir)"""


# watch video card
# watch -n0.1 nvidia-smi


# 20220929

# write data in a file.
"""file1 = open("myfile.txt","w")
L = ["This is Delhi \n","This is Paris \n","This is London \n"] 

# \n is placed to indicate EOL (End of Line)
file1.write("Hello \n")
file1.writelines(L)
file1.close() #to change file access modes
"""

from scipy import stats
# def give_ci(list, round=True):
#     mean = np.average(list)
#     std = np.std(list)
#     ci = stats.norm.interval(0.95, loc=mean, scale=std)
#     if round:
#         mean = np.round(mean, 2)
#         ci = (np.round(ci[0], 2), np.round(ci[1], 2))
#
#     return mean, ci


def give_ci(data, round_result=True):
    # Statistical tests
    shapiro_test = stats.shapiro(data)
    ks_test = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data, ddof=1)))

    # print("Shapiro-Wilk Test:")
    # print("Statistic:", shapiro_test.statistic)
    # print("p-value:", shapiro_test.pvalue)
    #
    # print("\nKolmogorov-Smirnov Test:")
    # print("Statistic:", ks_test.statistic)
    # print("p-value:", ks_test.pvalue)

    mean = np.mean(data)

    # Assuming normality if p-values are > 0.05
    if shapiro_test.pvalue > 0.05 and ks_test.pvalue > 0.05:

        std = np.std(data, ddof=1)
        ci = stats.norm.interval(0.95, loc=mean, scale=std / np.sqrt(len(data)))
        # print("\n95% Confidence Interval (Normal Assumption):", ci)
    else:
        # Bootstrap CI as an alternative
        n_bootstrap = 1000
        bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        ci = np.percentile(bootstrap_means, [2.5, 97.5])
        # print("\nBootstrap 95% Confidence Interval:", ci)

    # Rounding if required
    if round_result:
        mean = np.round(mean, 2)
        ci = (np.round(ci[0], 2), np.round(ci[1], 2))

    return mean, ci

# You can calculate the false positive rate and true positive rate associated to different threshold levels as follows:
# https://stackoverflow.com/questions/61321778/how-to-calculate-tpr-and-fpr-in-python-without-using-sklearn
def roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def mean_f_p_r(actual, predicted, best=10, pr_plot=False):
    list_f1 = []
    list_p = []
    list_r = []
    for r in range(len(actual)):
        y_actual = actual[r]
        y_predicted0 = predicted[r]
        y_predicted0 = [remove_space(item) for item in y_predicted0]
        y_predicted = []
        for item in y_predicted0:
            if item not in y_predicted:
                y_predicted.append(item)
        y_predicted = y_predicted[:best]
        y_score = 0

        for p, prediction in enumerate(y_predicted):
            if prediction in y_actual:  # and prediction not in y_predicted[:p]:
                y_score += 1

        if not y_predicted:
            y_p = 0
            y_r = 0
        else:
            y_p = y_score / len(y_predicted)
            y_r = y_score / len(y_actual)
        if y_p != 0 and y_r != 0:
            y_f1 = 2 * (y_p * y_r / (y_p + y_r))
        else:
            y_f1 = 0
        list_f1.append(y_f1)
        list_p.append(y_p)
        list_r.append(y_r)

    if pr_plot:
        return list_f1, list_p, list_r
    else:
        return np.mean(list_f1), np.mean(list_p), np.mean(list_r)


def read_excel(file):
    # read by default 1st sheet of an Excel file
    dataframe1 = pd.read_excel(file)  # sheet_name=0, [0, 1, "Sheet5"]: Load 1st, 2nd, sheet “Sheet5” as a dict
    return dataframe1

# 0630
# Removing duplicates from list: Using *set()
# It first removes the duplicates and returns a dictionary which has to be converted to list.
"""l = [1, 2, 4, 2, 1, 4, 5]
print("Original List: ", l)
res = [*set(l)]
print("List after removing duplicate elements: ", res)"""


# 20231002
def show_occur(search_term, input_text):
    # 输入文本
    # input_text = "这是一个示例文本，其中包含多次示例字段。示例字段可以出现多次。"

    # 要查找的字段
    # search_term = "示例字段"

    # 使用字符串的 find 方法来查找字段的位置
    start = 0
    while start < len(input_text):
        index = input_text.find(search_term, start)
        if index == -1:
            break
        print(f"字段 \"{search_term}\" 出现在位置 {index}")
        start = index + len(search_term)


# 20231012
def convert_date(input_date):
    # Input date string
    # input_date = "18-FEB-16"

    # Define a dictionary to map month abbreviations to their numeric values
    month_mapping = {
        'JAN': '01',
        'FEB': '02',
        'MAR': '03',
        'APR': '04',
        'MAY': '05',
        'JUN': '06',
        'JUL': '07',
        'AUG': '08',
        'SEP': '09',
        'OCT': '10',
        'NOV': '11',
        'DEC': '12'
    }

    # Extract components from the input date string
    date_components = input_date.split('-')  # .strip('[]').split('-')
    day = date_components[0]
    month = month_mapping[date_components[1]]
    year = '20' + date_components[2]

    # Create a datetime object
    date_object = datetime.strptime(year + month + day, '%Y%m%d')

    # Format the date as 'YYYYMMDD'
    formatted_date = date_object.strftime('%Y%m%d')

    # Print the result
    # print(formatted_date)
    return formatted_date


# 20231106
def bar_plot(categories, values):
    # 示例数据
    # categories = ['Category A', 'Category B', 'Category C', 'Category D']
    # values = [20, 35, 45, 27]

    # 创建柱状图
    plt.bar(categories, values)

    # 添加标题和标签
    # plt.title('柱状图示例')
    # plt.xlabel('类别')
    # plt.ylabel('值')

    # 显示图形
    plt.show()


# 20231201
def get_sen_spec(true_labels, predicted_labels):
    """sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)"""
    # True labels (ground truth)
    # true_labels = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    #
    # # Predicted labels (model's predictions)
    # predicted_labels = [1, 1, 1, 1, 0, 1, 0, 0, 0, 1]

    # Calculate True Positives (TP), False Negatives (FN), True Negatives (TN), and False Positives (FP)
    TP = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 1)
    FN = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 1 and p == 0)
    TN = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 0)
    FP = sum(1 for t, p in zip(true_labels, predicted_labels) if t == 0 and p == 1)

    # Calculate Sensitivity (True Positive Rate)
    sensitivity = TP / (TP + FN)

    # Calculate Specificity (True Negative Rate)
    specificity = TN / (TN + FP)
    # Print the results
    # print(f'Sensitivity (True Positive Rate): {sensitivity:.2f}')
    # print(f'Specificity (True Negative Rate): {specificity:.2f}')
    return sensitivity, specificity

# p_value two numbers part 1

# import numpy as np
# from scipy.stats import chi2_contingency
#
#
# def chi2_two_number(a, b):
#     """output
#     第一个值为卡方值，第二个值为P值，第三个值为自由度，第四个为与原数据数组同维度的对应理论值"""
#     local_d = np.array([[a, b], [249-a, 1055-b]])
#     chi, p, dof, _ = chi2_contingency(local_d)
#     if p < 0.01:
#         out_p = '<0.01'
#     elif p < 0.05:
#         out_p = '<0.05'
#     else:
#         out_p = str(np.round(p,4))
#     return chi, out_p, dof, _
#
#
#
# """
# 杀虫效果	甲	乙	丙
# 死亡数	37	49	23
# 未死亡数	150	100	57"""
#
# d = np.array([[37, 49, 23], [150, 100, 57]])
# # print(chi2_contingency(d))
#
# """
# group	      sarcopenic  non-sarcopenic
# Diabetes	        71                  109
# NoDiabetes	    178                 946
# """
#
# d2 = np.array([[71, 109], [178, 946]])
# # print(chi2_contingency(d2))
#
# # gender and race
# print('gender and race ---------------------------------------')
# input_text = """72		191
# 177		864
# 192		906
# 36		75
# 19		66
# 2		8"""
#
# collect0 = []
# collect1 = []
# collect2 = []
# collect3 = []
# for line in input_text.split('\n'):
#     line = line.split('		')
#     a = int(line[0])
#     b = int(line[1])
#
#     collect0.append(round((a+b)/13.04, 2))
#     collect1.append(round(a/2.49, 2))
#     collect2.append(round(b/10.55, 2))
#     collect3.append(chi2_two_number(a, b)[1])
#
# for collect in [collect0,collect1,collect2,collect3]:
#     for item in collect:
#         print(item)
#     print()
#
# # Specific Diagnosis
# print('Specific Diagnosis --------------------')
# collect4 = []
# input_text2 = """Diabetes both 180.0 13.8 71.0 28.51 109.0 10.33
# Dementia 7.0 0.54 5.0 2.01 2.0 0.19
# Cerebrovascular disease 52.0 3.99 19.0 7.63 33.0 3.13
# Hemiplegia or paraplegia 6.0 0.46 4.0 1.61 2.0 0.19
# Myocardial infarction 22.0 1.69 10.0 4.02 12.0 1.14
# Congestive heart failure 56.0 4.29 30.0 12.05 26.0 2.46
# Peripheral vascular disease 41.0 3.14 20.0 8.03 21.0 1.99
# Chronic pulmonary disease 210.0 16.1 61.0 24.5 149.0 14.12
# Peptic ulcer disease 12.0 0.92 6.0 2.41 6.0 0.57
# Liver disease both 78.0 5.98 28.0 11.24 50.0 4.74
# Rheumatic disease 43.0 3.3 14.0 5.62 29.0 2.75
# Renal disease 78.0 5.98 39.0 15.66 39.0 3.7
# AIDS/HIV 5.0 0.38 3.0 1.2 2.0 0.19
# Any malignancy 150.0 11.5 45.0 18.07 105.0 9.95
# Metastatic solid tumor 35.0 2.68 14.0 5.62 21.0 1.99
# Phalangeal 10 0.77 0 0.0 10 0.95
# Craniofacial 6 0.46 3 1.2 3 0.28
# Osteoporosis 135 10.35 30 12.05 105 9.95
# All Others (Excluding Craniofacial and Phalangeal) 92 7.06 25 10.04 67 6.35"""
#
# for line in input_text2.split('\n'):
#     line = line.split(' ')
#     a = float(line[-4])
#     b = float(line[-2])
#     # print(a,b)
#
#     collect4.append(chi2_two_number(a, b)[1])
#
# for collect in [collect4]:
#     for item in collect:
#         print(item)
#     print()
#
# cccccc
#
# input_text4 = """114
# 76
# 38
# 85
# 86
# 61
# 25
# 27
# 60"""
# for line in input_text4.split('\n'):
#
#     line = line
#     # print(line)
#     a = float(line)
#     print(round(a/2.49, 2))

# p_value two numbers part 2

# D:\PycharmProjects\sarcopenia_prediction\BiLSTM\code_0510_2021_2022\z_public_tools\demographic_table_0721.py
# import scipy.stats as stats
#
# # mannwhitneyu test
# def test(array_1, array_2):
#     d, p = stats.mannwhitneyu(array_1, array_2)
#     if p < 0.05:
#         # significance = '1'
#         print('p-value', p,'significant')
#     else:
#         # significance = ''
#         print('p-value', p)
#
# # overall number
# print('\npatients and non-patient', len(age_list_patient), len(age_list_nonpatient))
# print('all age mean', np.mean(age_list), 'age std', np.std(age_list))
# print('     patient age mean', np.mean(age_list_patient), 'std', np.std(age_list_patient))
# print('     non-patient age mean', np.mean(age_list_nonpatient), 'std', np.std(age_list_nonpatient))
#
# print('-'*50+'\nage significant')
# test(age_list_patient,age_list_nonpatient)