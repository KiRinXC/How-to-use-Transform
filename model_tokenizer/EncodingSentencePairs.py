# Details
# from transformers import AutoTokenizer
#
# checkpoint = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
#
# sequences = ["Are you Ok?", "I'm OK!"]
# inputs_direct = tokenizer("Are you Ok?", "I'm OK!")
# inputs_indirect = tokenizer(sequences)
# print(inputs_direct)
# print(inputs_indirect)
# print(tokenizer.convert_ids_to_tokens(inputs_direct["input_ids"]))

# Real-Use
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sentence1_list = ["First sentence.", "second sentence.", "Third one.","fourth one."]
sentence2_list = ["How are you?", "I am fine,", "thank you,","And you?"]

tokens = tokenizer(
    sentence1_list,
    sentence2_list,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
print(tokens)
print(tokens['input_ids'].shape)
