# -*-coding:utf-8 -*-

"""
# File       : main.py
# Time       ：2025/1/6 下午9:14
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：认证机制模型判断部分
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import logging


# 配置日志
logging.basicConfig(level=logging.INFO)  # 设置日志级别为 INFO
logger = logging.getLogger(__name__)  # 获取一个日志记录器


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


# FastAPI应用实例
app = FastAPI()


# 定义请求的Schema
class PredictionRequest(BaseModel):
    uid: int
    user_history: str
    current_text: str


# 定义HTTP API的预测接口
@app.post("/predict/")
async def predict_endpoint(request: PredictionRequest):
    try:
        # 编码输入
        encoded_input = encode_text(request.user_history, request.current_text, tokenizer)

        # 检查接收到的数据
        # print("variable:", request.uid, request.current_text)

        # 模型预测
        probability = predict(model, encoded_input, device)

        # 判断结果
        threshold = 0.19604818522930145
        result = 1 if probability > threshold else 0

        # 准备发送到目标地址的数据
        payload = {
            "uid": request.uid,
            "result": result,
            "content": request.current_text
        }

        # 使用 requests 发送 HTTP 请求
        target_url = "http://localhost:28080/Authentication/add"
        response = requests.post(target_url, json=payload, headers={"Content-Type": "application/json; charset=utf-8"})

        # 检查请求是否成功
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail=f"发送失败 {target_url}: {response.text}")

        # 返回目标地址的响应
        return {"detail": f"发送结果 {target_url}", "target_response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# 主函数
if __name__ == "__main__":
    # 加载BERT模型和Tokenizer
    tokenizer = BertTokenizer.from_pretrained("..//model//bert-base-chinese/")
    bert_model = BertModel.from_pretrained("..//model//bert-base-chinese")

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertBiGRUAttentionModel(bert_model)
    model.to(device)

    # 加载模型参数
    # model_load_path = '..//model//fine-tuning//fine-tuning-bert-BiGRU-ATT-1.pt'
    model_load_path = '..//model//fine-tuning//fine-tuning-bert-BiGRU-ATT-convergence-1.pt'
    state_dict = torch.load(model_load_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    # 设置模型为评估模式-确保模型在推理阶段的行为稳定和一致
    model.eval()

    uvicorn.run(app, host="0.0.0.0", port=28081)
