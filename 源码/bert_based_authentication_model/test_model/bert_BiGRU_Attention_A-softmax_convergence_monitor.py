# -*-coding:utf-8 -*-

"""
# File       : bert_BiGRU_Attention_A-softmax_convergence_monitor.py
# Time       ：2025/5/1 下午5:17
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：绘制训练集和测试集的损失，确定训练Bert+BiGRU+Attention+A-softmax模型至收敛所需的epoch
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import pandas as pd
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR


# 数据集划分
def split_dataset(df):
    train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
    return train_set, test_set


# 构建用户历史文本
def build_user_history(train_set):
    user_history = train_set.groupby('sender')['content'].apply(lambda x: ' '.join(x)).reset_index()
    user_history.columns = ['sender', 'history']
    return user_history


# 构造训练集
def construct_training_samples(train_set, user_history, negative_sample_count=3):
    all_users = train_set['sender'].unique()

    training_samples = []
    for user in all_users:
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


# 构造测试集
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


class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m=3):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m  # 角度间隔因子
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        # 初始化参数
        nn.init.xavier_normal_(self.weight)

        # 预计算cos(mθ)和sin(mθ)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # 确保cos(θ)非负的阈值
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m  # 用于调整边界

    def forward(self, x):
        # 归一化特征和权重
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        x_normalized = x / x_norm

        W_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        W_normalized = self.weight / W_norm

        # 计算余弦相似度
        cosine = x_normalized.mm(W_normalized.t())  # (batch_size, num_classes)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  # 确保数值稳定性

        # 应用角度间隔变换: cos(θ + m) = cosθ*cosm - sinθ*sinm
        phi = cosine * self.cos_m - sine * self.sin_m

        # 当cosθ > cos(π - m)时，使用调整后的cos值；否则保持原cosθ - m*sinθ
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 乘以特征模长（根据SphereFace设定，训练时可选择是否保留）
        output = phi * x_norm  # 若特征未归一化，保留模长缩放
        return output


# 定义Bert+BiGRU+Attention模型
class BertBiGRUAttentionModel(nn.Module):
    def __init__(self, bert_model, hidden_size=128, num_classes=2):
        super(BertBiGRUAttentionModel, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = AngleLinear(hidden_size * 2, num_classes)

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
    # input_text = f"[HISTORY] {history} [CURRENT] {current_text}"
    input_text = f"[CLS] {history} [SEP] {current_text} [SEP]"
    return tokenizer(input_text, padding=False, truncation=True, return_tensors=None)


# 批量编码数据
class TextDataset(Dataset):
    def __init__(self, dataframe, user_history, tokenizer, max_length=512, is_training=True):
        self.data = dataframe
        self.user_history = user_history
        self.tokenizer = tokenizer
        self.max_length = max_length  # 设置最大截断长度
        self.is_training = is_training  # 区分训练集/测试集

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.is_training:
            history = row['history']
        else:
            sender_history = self.user_history[self.user_history['sender'] == row['sender']]
            history = sender_history['history'].values[0]
        current_text = row['content']

        encoded = encode_text(history, current_text, self.tokenizer)
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'label': row['label']
        }


# 动态填充input_ids和attention_mask和列表转张量
def collate_fn(batch, tokenizer):
    input_ids = [torch.tensor(item['input_ids']) for item in batch]
    attention_masks = [torch.tensor(item['attention_mask']) for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    # 动态填充到当前batch最大长度
    padded_inputs = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_masks},
        padding=True,
        return_tensors='pt'
    )

    return {
        'input_ids': padded_inputs['input_ids'],
        'attention_mask': padded_inputs['attention_mask'],
        'labels': labels
    }


# 观察训练集和测试集
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


# 计算测试集损失
def evaluate_model(test_loader, model, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
    return total_loss / len(test_loader)


# 训练模型
def train_model(train_loader, test_loader, model, optimizer, device, epochs=10):
    # 交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    # 动态调整学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)

            # 计算损失
            loss = loss_fn(logits, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)
        avg_test_loss = evaluate_model(test_loader, model, device)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Test Loss: {avg_test_loss:.4f}")

    return train_losses, test_losses


# 绘制不同epoch对应的训练集损失和测试集损失
def plot_loss_curve(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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
    negative_sample_count = 3  # 设置负样本数量
    train_samples = construct_training_samples(train_set, user_history, negative_sample_count=negative_sample_count)

    # 构造测试集
    train_user_ids = train_set['sender'].unique()  # 记录训练集中的用户ID
    test_set = modify_test_set(test_set, train_user_ids)

    # 平衡测试集正负样本的数量
    test_set = balance_positive_negative_samples(test_set, train_user_ids)

    # 观察训练集和测试集
    analyze_dataset(train_set, test_set)

    # 批量编码训练集和测试集
    train_dataset = TextDataset(
        dataframe=train_samples,
        user_history=user_history,
        tokenizer=tokenizer,
        is_training=True
    )
    test_dataset = TextDataset(
        dataframe=test_set,
        user_history=user_history,
        tokenizer=tokenizer,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiGRUAttentionModel(bert_model)
    model.to(device)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 确定训练组合模型至收敛所需的epoch
    epochs = 3
    train_losses, test_losses = train_model(train_loader, test_loader, model, optimizer, device,
                                            epochs)

    plot_loss_curve(train_losses, test_losses)
