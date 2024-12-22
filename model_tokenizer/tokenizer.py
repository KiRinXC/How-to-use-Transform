from transformers import BertTokenizer

# 加载分词器并保存
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer.save_pretrained('../models/bert-base-cased')
#
# from transformers import AutoTokenizer
#
# tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
# tokenizer.save_pretrained('../models/bert-base-cased')

from transformers import AutoTokenizer
from transformers.data.data_collator import tolist

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')
sequence = "Using a Transformer network is simple"

# 简答分词
tokens = tokenizer.tokenize(sequence)
print(tokens)
#
# # 将分好的词转换成编码
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)
#
# # 直接通过序列生成编码
# sequence_ids = tokenizer.encode(sequence)
# print(sequence_ids)
#
# # 标准编码方式
# tokenized_text =tokenizer(sequence)
# print(tokenized_text)
#
#
# # 解码
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
#
# decode_string = tokenizer.decode([7993, 170, 13809, 23763, 2443, 1110, 3014])
# print(decode_string)
#
# decode_string = tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102])
# print(decode_string)