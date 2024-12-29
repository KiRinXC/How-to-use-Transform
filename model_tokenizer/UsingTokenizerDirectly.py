# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequences =["How are you?","Nice to meet you!"]
#
# model_inputs = tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")
# print(model_inputs)


# Padding
# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequences =["How are you?","Nice to meet you!"]
#
# model_inputs_1 = tokenizer(sequences,padding="longest")
# model_inputs_2 = tokenizer(sequences,padding="max_length")
# print(model_inputs_1)
# print(model_inputs_2)

#Truncation
# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequences =["How are you?","Nice to meet you!"]
#
# model_inputs = tokenizer(sequences, max_length=4, truncation=True)
# print(model_inputs)


# return_tensor
# from transformers import AutoTokenizer
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequences =["How are you?","Nice to meet you!"]
#
# print(tokenizer(sequences,padding=True, return_tensors="pt"))
# print(tokenizer(sequences,padding=True, return_tensors="np"))


from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!"
]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokens)
output = model(**tokens)
print(output.logits)