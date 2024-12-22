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


from transformers import AutoModelForSequenceClassification,AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = [
    "Using a Transformer network is simple",
    "Today is a good day!",
    "Amazing!"]

inputs = tokenizer(sequence, padding=True,return_tensors="pt")
print(inputs)
outputs = model(**inputs)
print(outputs)