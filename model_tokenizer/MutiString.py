# 先分词再编码，最后输入模型
# import torch
# from transformers import AutoModelForSequenceClassification,AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# sequence = "Using a Transformer network is simple"

# # 先分词
# tokens =tokenizer.tokenize(sequence)
# # 再编码
# ids =tokenizer.convert_tokens_to_ids(tokens)
# # 转成对应张量输入模型
# input_ids=torch.tensor([ids])
# print(input_ids)
#
# outputs = model(input_ids)
# print(outputs)


# 直接使用分词器
# from transformers import AutoModelForSequenceClassification,AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# sequence = "Using a Transformer network is simple"
# tokens = tokenizer(sequence_list, padding=True,return_tensors="pt")
#
# print(tokens)
#
# outputs = model(**tokens)
# print(outputs.logits)


# from transformers import AutoModelForSequenceClassification,AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# sequence = [
#     "Using a Transformer network is simple",
#     "Today is a good day!",
#     "Amazing!"]
#
# inputs = tokenizer(sequence, padding=True,return_tensors="pt")
# print(inputs)
# outputs = model(**inputs)
# print(outputs)


# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
#
# sequence1_ids = [[200, 200, 200]]
# sequence2_ids = [[200, 200]]
# batched_ids = [
#     [200, 200, 200],
#     [200, 200, tokenizer.pad_token_id],
# ]
# print(tokenizer.pad_token_id)
#
# print(model(torch.tensor(sequence1_ids)).logits)
# print(model(torch.tensor(sequence2_ids)).logits)
# print(model(torch.tensor(batched_ids)).logits)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]
batched_attention_masks = [
    [1, 1, 1],
    [1, 1, 0],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
outputs = model(
    torch.tensor(batched_ids),
    attention_mask=torch.tensor(batched_attention_masks))
print(outputs.logits)


