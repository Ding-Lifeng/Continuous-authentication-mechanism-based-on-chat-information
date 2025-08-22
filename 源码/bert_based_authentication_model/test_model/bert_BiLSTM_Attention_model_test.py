# -*-coding:utf-8 -*-

"""
# File       : bert_BiLSTM_Attention_model_test.py
# Time       ：2025/2/27 下午10:10
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：使用Bert+BiLSTM+Attention+softmax模型结构开展测试
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_curve, auc
from transformers import BertTokenizer, BertModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 定义Bert+BiLSTM+Attention模型
class BertBiLSTMAttentionModel(nn.Module):
    def __init__(self, bert_model, hidden_size=128, num_classes=2):
        super(BertBiLSTMAttentionModel, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # (batch_size, seq_len, 768)

        # BiLSTM处理
        lstm_output, _ = self.lstm(sequence_output)  # (batch_size, seq_len, hidden_size*2)

        # Attention机制
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size*2)

        # 分类
        logits = self.fc(weighted_output)  # (batch_size, num_classes)
        return logits


# 数据集划分
# def split_dataset(df):
#     train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
#     return train_set, test_set


# 平衡正负样本的数量
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


# 构建用户历史文本
def build_user_history(train_set):
    user_history = train_set.groupby('sender')['content'].apply(lambda x: ' '.join(x)).reset_index()
    user_history.columns = ['sender', 'history']
    return user_history


# 编码输入文本
def encode_text(history, current_text, tokenizer):
    # input_text = f"[HISTORY] {history} [CURRENT] {current_text}"
    input_text = f"[CLS] {history} [SEP] {current_text} [SEP]"
    return tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")


# 预测函数
def predict(model, encoded_input, device):
    input_ids = encoded_input['input_ids'].to(device)
    attention_mask = encoded_input['attention_mask'].to(device)

    # 模型推断-softmax层输出分类结果
    output = model(input_ids, attention_mask)
    probabilities = torch.softmax(output, dim=1)
    return probabilities[0, 1].item()  # 返回属于当前用户的概率


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


# 测试集评估
def evaluate_model(test_set, user_history, model, tokenizer, device):
    y_true = []
    y_scores = []

    # 生成每条测试样本的预测概率
    for _, row in test_set.iterrows():
        history = user_history[user_history['sender'] == row['sender']]['history'].values
        if len(history) == 0:
            continue  # 跳过没有历史的样本
        encoded_input = encode_text(history[0], row['content'], tokenizer)
        score = predict(model, encoded_input, device)
        y_scores.append(score)
        y_true.append(int(row['label']))

    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # 计算最佳阈值
    optimal_idx = (tpr - fpr).argmax()  # 找到TPR - FPR最大值的索引
    optimal_threshold = thresholds[optimal_idx]

    # 根据分类阈值和预测概率判断样本标签
    y_pred = [1 if s > optimal_threshold else 0 for s in y_scores]

    return fpr, tpr, roc_auc, optimal_threshold, y_true, y_pred


# 交叉验证测试模型
def cross_validate(df, model, tokenizer, device, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    colors = plt.colormaps.get_cmap('tab10')

    auc_list = []
    threshold_list = []
    all_y_true = []
    all_y_pred = []

    plt.figure(figsize=(8, 6))

    for fold, (train_index, test_index) in enumerate(kf.split(df)):
        train_set = df.iloc[train_index].copy()
        test_set = df.iloc[test_index].copy()

        # 构造用户历史文本
        user_history = build_user_history(train_set)

        # 构造测试集
        train_user_ids = train_set['sender'].unique()
        test_set = modify_test_set(test_set, train_user_ids)

        # 平衡测试集正负样本的数量
        test_set = balance_positive_negative_samples(test_set, train_user_ids)

        fpr, tpr, roc_auc, optimal_thresh, y_true, y_pred = evaluate_model(
            test_set, user_history, model, tokenizer, device
        )

        auc_list.append(roc_auc)
        threshold_list.append(optimal_thresh)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        # 绘制ROC曲线
        plt.plot(fpr, tpr, lw=2, color=colors(fold % 10), label=f'Fold {fold + 1} (AUC = {roc_auc:.2f})')

    # 绘图及总结
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    mean_auc = np.mean(auc_list)
    mean_threshold = np.mean(threshold_list)
    plt.title(f'Cross Validate')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()

    print(f"\n平均AUC: {mean_auc:.4f}")
    print(f"平均最佳阈值: {mean_threshold:.4f}")
    print("\n平均分类报告（基于所有折的预测）：")
    print(classification_report(all_y_true, all_y_pred, digits=4))


# 主函数
if __name__ == "__main__":
    df = pd.read_csv('..\\..\\..\\数据集\\new_sms_zh.csv').astype(str)

    # 加载BERT模型和Tokenizer
    tokenizer = BertTokenizer.from_pretrained("..//..//model//bert-base-chinese/")
    bert_model = BertModel.from_pretrained("..//..//model//bert-base-chinese")

    # 分割数据集
    # train_set, test_set = split_dataset(df)

    # 构造用户历史文本
    # user_history = build_user_history(train_set)

    # 构造测试集
    # train_user_ids = train_set['sender'].unique()  # 记录训练集中的用户ID
    # test_set = modify_test_set(test_set, train_user_ids)

    # 平衡测试集正负样本的数量
    # test_set = balance_positive_negative_samples(test_set, train_user_ids)

    # 观察训练集和测试集
    # analyze_dataset(train_set, test_set)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiLSTMAttentionModel(bert_model)
    model.to(device)

    # 加载模型参数
    # model_load_path = '..//..//model//fine-tuning//fine-tuning-bert-BiLSTM-ATT-1.pt'
    model_load_path = '..//..//model//fine-tuning//fine-tuning-bert-BiLSTM-ATT-convergence-1.pt'
    state_dict = torch.load(model_load_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 设置模型为评估模式-确保模型在推理阶段的行为稳定和一致
    model.eval()

    # 测试并评估模型
    cross_validate(df, model, tokenizer, device)
