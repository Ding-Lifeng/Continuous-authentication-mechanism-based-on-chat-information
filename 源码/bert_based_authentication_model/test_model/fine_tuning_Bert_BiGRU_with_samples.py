# -*-coding:utf-8 -*-

"""
# File       : fine_tuning_Bert_BiGRU_with_samples.py
# Time       ：2025/2/27 下午9:00
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：使用训练集微调组合模型(Bert-BiGRU)
"""

import random
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import pandas as pd
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


# 构造微调的训练集
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


# 定义Bert+BiGRU模型
class BertBiGRUModel(nn.Module):
    def __init__(self, bert_model, hidden_size=128, num_classes=2):
        super(BertBiGRUModel, self).__init__()
        self.bert = bert_model
        self.gru = nn.GRU(input_size=768, hidden_size=hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # (batch_size, seq_len, 768)

        # BiGRU处理
        gru_output, _ = self.gru(sequence_output)  # (batch_size, seq_len, hidden_size*2)

        # 取GRU输出的最后一个时间步的隐藏状态（序列的综合信息）作为输入
        last_hidden_state = gru_output[:, -1, :]

        # 分类
        logits = self.fc(last_hidden_state)  # (batch_size, num_classes)
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


# 观察训练集
def analyze_dataset(train_set):
    # 统计训练集信息
    train_total = len(train_set)
    train_users = train_set['sender'].nunique()

    # 输出统计信息
    print("训练集统计信息:")
    print(f"- 数据总量: {train_total}")
    print(f"- 用户数量: {train_users}")


# 训练模型
def train_model(train_loader, model, optimizer, device, epochs=1):
    model.train()
    # 交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    # 动态调整学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
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

        scheduler.step()


# 主函数
if __name__ == "__main__":
    df = pd.read_csv('..\\..\\..\\数据集\\new_sms_zh.csv').astype(str)

    # 加载BERT模型和Tokenizer
    tokenizer = BertTokenizer.from_pretrained("..//..//model//bert-base-chinese/")
    bert_model = BertModel.from_pretrained("..//..//model//bert-base-chinese")

    # 分割数据集
    train_set, test_set = split_dataset(df)

    # 构造用户历史文本
    user_history = build_user_history(train_set)

    # 构造训练样本
    negative_sample_count = 3  # 设置负样本数量
    train_samples = construct_training_samples(train_set, user_history, negative_sample_count=negative_sample_count)

    # 观察训练集
    analyze_dataset(train_set)

    # 批量编码训练集
    train_dataset = TextDataset(
        dataframe=train_samples,
        user_history=user_history,
        tokenizer=tokenizer,
        is_training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiGRUModel(bert_model)
    model.to(device)

    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # 模型训练
    epochs = 1
    train_model(train_loader, model, optimizer, device, epochs)

    # 保存微调后的模型
    # model_save_path = '..//..//model//fine-tuning//fine-tuning-bert-BiGRU-1.pt'
    model_save_path = '..//..//model//fine-tuning//fine-tuning-bert-BiGRU-convergence-1.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"模型已保存至 {model_save_path}")
