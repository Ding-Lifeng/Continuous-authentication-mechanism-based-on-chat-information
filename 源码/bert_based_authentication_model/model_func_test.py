# -*-coding:utf-8 -*-

"""
# File       : model_func_test.py
# Time       ：2025/1/7 下午12:01
# Author     ：丁笠峰
# version    ：python 3.9
# note       ：测试Bert模型功能
"""

from transformers import BertTokenizer, BertForSequenceClassification

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('..//model//bert-base-chinese/')
model = BertForSequenceClassification.from_pretrained('..//model//bert-base-chinese')

# 示例文本
text = "云南的气候是什么样的？"

# 对文本进行编码
inputs = tokenizer(text, return_tensors="pt")

# 推理
outputs = model(**inputs)

# 获取模型的输出
logits = outputs.logits
print(logits)
