
# 加载模型分词器

# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "Amazing",
# ]
# inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# print(inputs)


# 加载预训练模型

# from transformers import AutoModel
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)


# 模型输出维度
# from transformers import AutoTokenizer, AutoModel
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModel.from_pretrained(checkpoint)
#
# raw_inputs = [
#     "I've been waiting for a HuggingFace course my whole life.",
#     "Amazing",
# ]
# inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)
# print(outputs[0])
# print(outputs["last_hidden_state"])


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "Amazing",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.logits)
print(outputs.logits.shape)


# 将logits值转换成概率值
import torch
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)

predict_tolist = predictions.detach().numpy()
print(predict_tolist)


# 再对输出稍加处理
input_list=[]
for sample in predict_tolist:
    dict ={"label":None,"score":None}
    if sample[0]>sample[1]:
        dict["label"]=model.config.id2label[0]
        dict["score"]=round(sample[0], 4)
    else:
        dict["label"]=model.config.id2label[1]
        dict["score"]=round(sample[1], 4)
    input_list.append(dict)

# 得到标准输出格式
print(input_list)
# [{'label': 'POSITIVE', 'score': 0.9598}, {'label': 'POSITIVE', 'score': 0.9999}]