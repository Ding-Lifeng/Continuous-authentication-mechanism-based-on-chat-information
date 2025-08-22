# -*-coding:utf-8 -*-

"""
# File       : locust_testing.py
# Time       ：2025/5/5 下午16:06
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：使用Locust进行用户认证模型的吞吐量和响应时间测试
"""


from locust import HttpUser, task, between
import pandas as pd
import random
import json


# 构建用户历史文本
def build_user_history(train_set):
    user_history = train_set.groupby('sender')['content'].apply(lambda x: ' '.join(x)).reset_index()
    user_history.columns = ['sender', 'history']
    return user_history


# 加载数据集
df = pd.read_csv('..\\..\\数据集\\new_sms_zh.csv').astype(str)
user_history_df = build_user_history(df)

# 创建测试样本列表
test_data = []
for _, row in df.iterrows():
    sender = row['sender']
    current_text = row['content']
    history_row = user_history_df[user_history_df['sender'] == sender]
    test_data.append({
        "uid": random.randint(1000, 9999),
        "user_history": history_row.iloc[0]['history'],
        "current_text": current_text
    })


class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def send_predict_request(self):
        sample = random.choice(test_data)
        headers = {"Content-Type": "application/json; charset=UTF-8"}
        self.client.post("/predict/", data=json.dumps(sample), headers=headers)
