# -*-coding:utf-8 -*-

"""
# File       : SVM_test.py
# Time       ：2025/2/26 下午9:22
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：使用SVM开展测试
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
import random
import jieba
import numpy as np


# 数据集划分
# def split_dataset(df):
#     train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
#     return train_set, test_set


# 修改测试集生成负样本
def modify_test_set(test_set, train_user_ids):
    def modify(row):
        if row['sender'] not in train_user_ids:
            row['sender'] = random.choice(train_user_ids)  # 修改为训练集中任意用户ID
            row['label'] = 0  # 标记为负样本
        else:
            row['label'] = 1  # 未修改，标记为正样本
        return row

    return test_set.apply(modify, axis=1)


# 平衡测试集正负样本的数量
def balance_positive_negative_samples(test_set, train_user_ids):
    # 统计正负样本数量
    positive_samples = test_set[test_set['label'] == 1]
    negative_samples = test_set[test_set['label'] == 0]

    # 如果正样本数量大于负样本数量
    if len(positive_samples) > len(negative_samples):
        # 计算需要修改的样本数量
        excess_positive_count = (len(positive_samples) - len(negative_samples)) // 2

        # 随机选择需要修改的正样本
        positive_to_modify = positive_samples.sample(n=excess_positive_count, random_state=42)

        # 修改正样本为负样本
        def modify_to_negative(row):
            valid_ids = [uid for uid in train_user_ids if uid != row['sender']]
            row['sender'] = random.choice(valid_ids)  # 修改为训练集中任意其他用户ID
            row['label'] = 0  # 标注为负样本
            return row

        modified_positive_samples = positive_to_modify.apply(modify_to_negative, axis=1)

        # 更新测试集
        test_set.update(modified_positive_samples)

    return test_set


# 中文分词函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))


# SVM模型训练与评估
def train_and_evaluate_svm(train_set, test_set, return_predictions=False):
    # 使用TfidfVectorizer初始化
    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=None)

    # 存储预测结果和真实标签
    y_true = []
    y_pred = []

    # 针对测试集中的每一条数据
    for _, test_row in test_set.iterrows():
        sender = test_row['sender']
        content = test_row['content']

        # 提取训练集中的所有对应作者的content并标记为正样本
        positive_samples = train_set[train_set['sender'] == sender]
        num_positive_samples = len(positive_samples)

        # 提取训练集中的所有其他作者的content，并标记为负样本
        negative_samples = train_set[train_set['sender'] != sender]

        # 随机从训练集中的其他发送者中抽取与正样本数量相同的负样本
        negative_samples = negative_samples.sample(n=num_positive_samples, random_state=42)

        # 将正样本与负样本合并
        all_samples = pd.concat([positive_samples, negative_samples])

        # 将所有样本的content转化为特征向量并加入训练数据
        X_all = vectorizer.fit_transform(all_samples['content'])

        # 为每个样本标记标签
        y_all = np.concatenate([np.ones(num_positive_samples), np.zeros(num_positive_samples)])

        # 创建并训练支持向量机模型
        svm_model = SVC(kernel='linear')
        svm_model.fit(X_all, y_all)

        # 使用模型对当前测试样本进行预测
        X_test = vectorizer.transform([content])  # 将测试样本转换为特征向量
        y_pred_sample = svm_model.predict(X_test)  # 预测当前测试样本的标签

        # 记录真实标签和预测标签
        y_true.append(test_row['label'])
        y_pred.append(y_pred_sample[0])

    if return_predictions:
        return y_true, y_pred

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)

    # 计算精确率、召回率和F1值
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 输出评估指标
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')


def cross_validate_svm(df, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []

    fold = 1
    for train_idx, test_idx in kf.split(df):
        train_set = df.iloc[train_idx].copy()
        test_set = df.iloc[test_idx].copy()

        train_user_ids = train_set['sender'].unique()
        test_set = modify_test_set(test_set, train_user_ids)
        test_set = balance_positive_negative_samples(test_set, train_user_ids)

        print(f"\n Fold {fold}:")
        analyze_dataset(train_set, test_set)

        # 训练并评估模型，返回预测结果
        y_true, y_pred = train_and_evaluate_svm(train_set, test_set, return_predictions=True)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        fold += 1

    accuracy = accuracy_score(all_y_true, all_y_pred)
    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)

    print("\n===== Cross Validation Result =====")
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(f'Precision: {precision * 100:.2f}%')
    print(f'Recall: {recall * 100:.2f}%')
    print(f'F1 Score: {f1 * 100:.2f}%')


# 观察训练集和测试
def analyze_dataset(train_set, test_set):
    # 统计训练集信息
    train_total = len(train_set)
    train_users = train_set['sender'].nunique()

    # 统计测试集信息
    test_total = len(test_set)
    test_users = test_set['sender'].nunique()
    test_positive = len(test_set[test_set['label'] == 1])
    test_negative = len(test_set[test_set['label'] == 0])

    # 输出统计信息
    print("训练集统计信息:")
    print(f"- 数据总量: {train_total}")
    print(f"- 用户数量: {train_users}")
    print("\n测试集统计信息:")
    print(f"- 数据总量: {test_total}")
    print(f"- 用户数量: {test_users}")
    print(f"- 正样本数量: {test_positive}")
    print(f"- 负样本数量: {test_negative}")


# 主函数
if __name__ == "__main__":
    df = pd.read_csv('..\\..\\..\\数据集\\new_sms_zh.csv').astype(str)

    # 分割数据集
    # train_set, test_set = split_dataset(df)

    # 修改测试集生成负样本
    # train_user_ids = train_set['sender'].unique()  # 记录训练集中的用户ID
    # test_set = modify_test_set(test_set, train_user_ids)

    # 平衡测试集正负样本的数量
    # test_set = balance_positive_negative_samples(test_set, train_user_ids)

    # 观察训练集和测试集
    # analyze_dataset(train_set, test_set)

    # 训练与评估SVM模型
    # train_and_evaluate_svm(train_set, test_set)
    cross_validate_svm(df)
