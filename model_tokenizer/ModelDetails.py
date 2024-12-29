from transformers import BertModel
from transformers import AutoModel

# model = BertModel.from_pretrained('bert-base-cased')

# 加载模型
# model = BertModel.from_pretrained('../models/bert-base-cased')


#保存模型
model = AutoModel.from_pretrained('bert-base-cased')
model.save_pretrained('../models/bert-base-cased')
