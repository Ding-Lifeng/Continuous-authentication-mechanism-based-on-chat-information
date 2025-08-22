# -*-coding:utf-8 -*-

"""
# File       : fine_tuning_model_with_samples.py
# Time       ：2025/1/7 下午10:47
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：使用训练集微调组合模型(Bert-BiGRU-Attention)；该方案存在缺陷，未训练模型至收敛，
使用fine_tuning_model_to_convergence中的方案进行替代
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import pandas as pd


# 数据集划分
def split_dataset(df):
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
    return train_set, test_set


# 构建用户历史文本
def build_user_history(train_set):
    user_history = train_set.groupby('sender')['content'].apply(lambda x: ' '.join(x)).reset_index()
    user_history.columns = ['sender', 'history']
    return user_history


# 构造微调的训练集
def construct_training_samples(train_set, user_history, negative_sample_count=3):
    # 随机抽取训练集中一半的用户
    all_users = train_set['sender'].unique()
    selected_users = random.sample(list(all_users), len(all_users) // 2)

    training_samples = []
    for user in selected_users:
        # 获取正样本
        user_rows = train_set[train_set['sender'] == user]
        user_history_text = user_history[user_history['sender'] == user]['history'].values[0]

        for _, row in user_rows.iterrows():
            training_samples.append({
                'history': user_history_text,
                'content': row['content'],
                'label': 1  # 正样本
            })

            # 随机抽取负样本
            negative_samples = []
            available_users = [u for u in all_users if u != user]
            negative_users = random.sample(available_users, negative_sample_count)
            for neg_user in negative_users:
                # 从选中的负样本用户历史中获取内容
                neg_history = user_history[user_history['sender'] == neg_user]['history'].values[0]
                negative_samples.append({
                    'history': neg_history,
                    'content': row['content'],
                    'label': 0  # 负样本
                })

                # 添加负样本
            training_samples.extend(negative_samples)

    return pd.DataFrame(training_samples)


# 定义Bert+BiGRU+Attention模型
class BertBiGRUAttentionModel(nn.Module):
    def __init__(self, bert_model, hidden_size=128, num_classes=2):
        super(BertBiGRUAttentionModel, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # (batch_size, seq_len, 768)

        # BiGRU处理
        gru_output, _ = self.gru(sequence_output)  # (batch_size, seq_len, hidden_size*2)

        # Attention机制
        attention_weights = F.softmax(self.attention(gru_output), dim=1)  # (batch_size, seq_len, 1)
        weighted_output = torch.sum(attention_weights * gru_output, dim=1)  # (batch_size, hidden_size*2)

        # 分类
        logits = self.fc(weighted_output)  # (batch_size, num_classes)
        return logits


# 编码输入文本
def encode_text(history, current_text, tokenizer):
    input_text = f"[HISTORY] {history} [CURRENT] {current_text}"
    return tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")


# 训练模型
def train_model(train_samples, model, tokenizer, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    # epoch = 1
    for _, row in train_samples.iterrows():
        # 编码文本
        encoded_input = encode_text(row['history'], row['content'], tokenizer)
        input_ids = encoded_input['input_ids'].to(device)
        attention_mask = encoded_input['attention_mask'].to(device)
        label = torch.tensor([row['label']], dtype=torch.long).to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)

        # 计算损失
        loss = loss_fn(logits, label)
        loss.backward()
        optimizer.step()


# 主函数
if __name__ == "__main__":
    df = pd.read_csv('..\\..\\数据集\\new_sms_zh.csv').astype(str)

    # 加载BERT模型和Tokenizer
    tokenizer = BertTokenizer.from_pretrained("..//model//bert-base-chinese/")
    bert_model = BertModel.from_pretrained("..//model//bert-base-chinese")

    # 分割数据集
    train_set, test_set = split_dataset(df)

    # 构造用户历史文本
    user_history = build_user_history(train_set)

    # 构造训练样本
    negative_sample_count = 5  # 设置负样本数量
    train_samples = construct_training_samples(train_set, user_history, negative_sample_count=negative_sample_count)

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiGRUAttentionModel(bert_model)
    model.to(device)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 模型训练
    train_model(train_samples, model, tokenizer, optimizer, device)

    # 保存微调后的模型
    model_save_path = '..//model//fine-tuning//fine-tuning-bert-BiGRU-ATT-1.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")
