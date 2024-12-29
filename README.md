## 第一章：Transforms快速入门
### 1. 安装依赖  
- 首先安装 `requirements.txt` 中的所有依赖包  
```bash  
pip install -r requirements.txt
```  
- 若使用*科学上网*，需要降低 `requset` 和 `urllibs3` 两个依赖包的版本，否则在后续下载模型时会因网络代理而报错。
```bash
pip install requests==2.27.1
pip install urllib3==1.25.11  
```
### 2. 开箱即用的pipelines  

- `Transformers`库最基础的对象就是 `pipeline()` 函数，它封装了预训练模型和对应的前处理和后处理环节。**相当于是一个已经训练好能直接使用的工具**。只需输入文本，就能得到预期的答案。
- 常用的 `pipelines` 有以下几点：
	- `feature-extraction` （获得文本的向量化表示）
	- `fill-mask` （填充被遮盖的词、片段）
	- `ner`（命名实体识别）
	- `question-answering` （自动问答）
	- `sentiment-analysis` （情感分析）
	- `summarization` （自动摘要）
	- `text-generation` （文本生成）
	- `translation` （机器翻译）
	- `zero-shot-classification` （零训练样本分类）
- 在初次使用不同pipelines完成任务时，需要自动下载模型至C盘的 _.cache_ 处，因此需要耗费些许时间。
#### 2.1 情感分析
- 通过 `pipeline` 函数指定任务类型为：**sentiment-analysis**
```python
# 若不使用科学上网，则可选择镜像网站下载模型
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import pipeline  
  
classifier = pipeline("sentiment-analysis")  
result = classifier("Today is a good day,but I don't like it.")  
print(result)  
results_zh = classifier(  
  ["今天很好，但我不喜欢", "今天不好"]  
)  
results_en = classifier(  
    ["Today is good, but I don't like it", "Today is not very good"]  
)  
print(results_zh)  
print(results_en)
```
- `result` 是一个列表，包含 `label` 和 `score` 两个关键词。可见，模型对中英文都有效。
```txt
No model was supplied, defaulted to distilbert/distilbert-base-uncased-finetuned-sst-2-english and revision 714eb0f (https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
[{'label': 'NEGATIVE', 'score': 0.974997878074646}]
[{'label': 'NEGATIVE', 'score': 0.9165080785751343}, {'label': 'NEGATIVE', 'score': 0.6466353535652161}]
[{'label': 'NEGATIVE', 'score': 0.8463599681854248}, {'label': 'NEGATIVE', 'score': 0.9997707009315491}]
```

> pipeline 会自动选择模型来完成对应任务，如在情感分析中，默认选择微调好的英文情感模型 _distilbert-base-uncased-finetuned-sst-2-english_。


#### 2.2 零训练样本分类
- 指定任务类型为 `zero-shot-classification`，在 `classifier` 中输入待分类文本和候选标签。
```python
from transformers import pipeline  
  
classifier = pipeline("zero-shot-classification")  
result = classifier(  
"This is a course about the Transformers library",  
candidate_labels=["education", "politics", "business"],  
)  
print(result)
```
- 使用的模型是 _facebook/bart-large-mnli_
```txt
No model was supplied, defaulted to facebook/bart-large-mnli and revision d7645e1 (https://huggingface.co/facebook/bart-large-mnli).

{'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.8445993065834045, 0.11197393387556076, 0.043426718562841415]}
```

#### 2.3 文本生成
- 指定任务类型为 `text-generation` ，也可以指定生成的序列数和生成的最大长度。
```python
from transformers import pipeline  
  
generator = pipeline("text-generation")  
results = generator("今日吾虽死，")  
print(results)  
results = generator(  
    "In this course, we will teach you how to",  
    num_return_sequences=2,  
    max_length=20  
)  
print(results)
```
- 使用的模型是 _gpt2_，显然该模型缺少生成中文文本的能力
```text
No model was supplied, defaulted to openai-community/gpt2 and revision 607a30d (https://huggingface.co/openai-community/gpt2).

[{'generated_text': '今日吾虽死，我吾處－ ２？\n\n"But I didn\'t mean to mention you or anything, I wanted to tell everybody how my'}]
[{'generated_text': 'In this course, we will teach you how to understand the role of the law in the U.'}, {'generated_text': 'In this course, we will teach you how to use the power of intuition to understand that time and'}]

```
- 也可以在 [Model Hub](https://huggingface.co/models) 页面左边选择 [Text Generation](https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads) tag 查询支持的模型。例如，在相同的 pipeline 中加载 [distilgpt2](https://huggingface.co/distilgpt2) 模型：
```python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=20,
    num_return_sequences=1,
)
print(results)
```

```txt
[{'generated_text': 'In this course, we will teach you how to practice and gain experience in a few basic and highly'}]
```

- 文本生成任务中包含许多模型，例如，专门用于生成中文古诗的 [gpt2-chinese-poem](https://huggingface.co/uer/gpt2-chinese-poem) 模型，可以进入模型详情页学习模型如何使用。
```python
from transformers import pipeline

generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results = generator(
    "[CLS]梅 山 如 积 翠 ，",
    max_length=20,
    num_return_sequences=1,
)
print(results)
```

```txt
[{'generated_text': '[CLS]梅 山 如 积 翠 ， 湖 波 冷 日 初 黄 。 江 东 地 极 天 尽 处 ， 望 见 江 城 城 一 方 。 念 金 陵 已 无 家 ，'}]
```


#### 2.4 掩盖词填充
- 在给定一段部分词语被遮盖掉 (mask) 的文本，使用预训练模型来预测能够填充这些位置的词语。用  **< mask >** 来表示需要遮盖的文本，用 `top_k` 控制生成的序列数量。
```python
from transformers import pipeline  
  
unmasker = pipeline("fill-mask")  
results = unmasker("This course will teach you all about <mask> models.", top_k=2)  
print(results)
```
- 默认使用的模型是 _[distilroberta-base](https://huggingface.co/distilbert/distilroberta-base)_，这个模型只能应对句子中只有一个 mask 的情况。
```txt
No model was supplied, defaulted to distilbert/distilroberta-base and revision fb53ab8 (https://huggingface.co/distilbert/distilroberta-base).
[{'score': 0.1961977779865265, 'token': 30412, 'token_str': ' mathematical', 'sequence': 'This course will teach you all about mathematical models.'}, 
{'score': 0.04052717983722687, 'token': 38163, 'token_str': ' computational', 'sequence': 'This course will teach you all about computational models.'}]
```



#### 2.5 命名实体识别
- 命名实体识别负责从文本中抽取出指定类型的实体，例如人物、地点、组织等等
```python
from transformers import pipeline  
  
ner = pipeline("ner", grouped_entities=True)  
results = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")  
print(results)
```
- 模型正确地识别出了 Sylvain 是一个人物(PER)，Hugging Face 是一个组织(ORG)，Brooklyn 是一个地名(LOC)。
```txt
No model was supplied, defaulted to dbmdz/bert-large-cased-finetuned-conll03-english and revision 4c53496 (https://huggingface.co/dbmdz/bert-large-cased-finetuned-conll03-english).

[{'entity_group': 'PER', 'score': 0.9981694, 'word': 'Sylvain', 'start': 11, 'end': 18}, {'entity_group': 'ORG', 'score': 0.9796019, 'word': 'Hugging Face', 'start': 33, 'end': 45}, {'entity_group': 'LOC', 'score': 0.9932106, 'word': 'Brooklyn', 'start': 49, 'end': 57}]
```

#### 2.6 自动问答
- 通过给定的上下文回答问题，类似于阅读理解。
```python
from transformers import pipeline  
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')  
  
context = r"My name is Sylvain and I work at Hugging Face in Brooklyn"  
  
result = question_answerer(  
    question="Where do I work?",  
    context=context)  
print(result)
```
- 这里的自动问答实际上是一个**抽取式问答**模型，即从给定的上下文中抽取答案，而不是生成答案。
```txt
{'score': 0.6949771046638489, 'start': 33, 'end': 45, 'answer': 'Hugging Face'}
```

> 根据形式的不同，自动问答 (QA) 系统可以分为三种：

- **抽取式 QA (extractive QA)：**假设答案就包含在文档中，因此直接从文档中抽取答案；
- **多选 QA (multiple-choice QA)：**从多个给定的选项中选择答案，相当于做阅读理解题；
- **无约束 QA (free-form QA)：**直接生成答案文本，并且对答案文本格式没有任何限制。

#### 2.7 自动摘要
- 将长文本压缩成短文本，并且还要尽可能保留原文的主要信息。
```python
from transformers import pipeline  
  
summarizer = pipeline("summarization")  
results = summarizer(  
    """  
    America has changed dramatically during recent years. Not only has the number of    graduates in traditional engineering disciplines such as mechanical, civil,   
    electrical, chemical, and aeronautical engineering declined, but in most of   
    the premier American universities engineering curricula now concentrate on   
    and encourage largely the study of engineering science. As a result, there   
    are declining offerings in engineering subjects dealing with infrastructure,   
    the environment, and related issues, and greater concentration on high   
    technology subjects, largely supporting increasingly complex scientific   
    developments. While the latter is important, it should not be at the expense   
    of more traditional engineering.  
  
    Rapidly developing economies such as China and India, as well as other    industrial countries in Europe and Asia, continue to encourage and advance   
    the teaching of engineering. Both China and India, respectively, graduate   
    six and eight times as many traditional engineers as does the United States.   
    Other industrial countries at minimum maintain their output, while America   
    suffers an increasingly serious decline in the number of engineering graduates   
    and a lack of well-educated engineers.  
    """)  
print(results)
```

```txt
[{'summary_text': ' America has changed dramatically during recent years . The number of engineering graduates in the U.S. has declined in traditional engineering disciplines such as mechanical, civil, electrical, chemical, and aeronautical engineering . Rapidly developing economies such as China and India, as well as other industrial countries in Europe and Asia, continue to encourage and advance engineering .'}]
```

>通过pipelines可以轻松调用模型来应对各种任务，在尝试的过程中可以多参考 [Model Hub](https://huggingface.co/models) ，学习各种模型的基本使用。

### 3. pipelines知其所以然
- 这些简单易用的 pipeline 模型实际上封装了许多操作，以第一个情感分析 pipeline 为例，从输入的文本到模型输出的 _label_ 和 _score_ ，要经过三个步骤：
	1. 预处理 (preprocessing)，将原始文本转换为模型可以接受的输入格式；
	2. 将处理好的输入送入模型；
	3. 对模型的输出进行后处理 (postprocessing)，将其转换为人类方便阅读的格式。

---

#### 3.1 预处理
- 我们会使用每个模型对应的分词器 (tokenizer)来进行原始文本的预处理，具体来说就是
	1. 将输入切分为词语、子词或者符号（例如标点符号），统称为 **tokens**；
	2. 根据模型的词表将每个 token 映射到对应的 token 编号（即数字）；
	3. ~~根据模型的需要，可能会添加一些额外的输入（如生成中文古诗中的 [CLS]）。~~

- 每个模型都有特定的预处理操作，因此我们要使用与模型一致的预处理操作。这里使用 `AutoTokenizer` 类和它的 `from_pretrained()` 函数，它可以自动根据模型 checkpoint 名称来加载对应的分词器。

- 情感分析 pipeline 的默认 checkpoint 是 [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)，下面手工加载该模型的分词器进行分词：
```python
from transformers import AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
raw_inputs = [  
    "I've been waiting for a HuggingFace course my whole life.",  
    "Amazing",  
]  
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  
print(inputs)
```
- 简单提一点，在 tokenizer 中的参数中，padding是填充，truncation为截断，return_tensors是返回的张量类型（pt即Pytorch）
```txt
{
	'input_ids': tensor([
	[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
	          2607,  2026,  2878,  2166,  1012,   102],
	[  101,  6429,   102,     0,     0,     0,     0,     0,     0,     0,
	             0,     0,     0,     0,     0,     0]
	]), 
	'attention_mask': tensor([
	[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
	])
}
```

- 输出中包含两个键 `input_ids` 和 `attention_mask`，其中 `input_ids` 对应分词之后的 tokens 映射到的数字编号列表，而 `attention_mask` 则是用来标记哪些 tokens 是被填充的（这里“1”表示是原文，“0”表示是填充字符）。

> 可见，单词与token之间并非是一一映射的。

#### 3.2 将处理好的输入送入模型
- 预训练模型的加载方式和分词器 (tokenizer) 类似，Transformers 包提供了一个 `AutoModel` 类和对应的 `from_pretrained()` 函数。下面手工加载这个 distilbert-base 预训练模型：
```python
from transformers import AutoModel  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
model = AutoModel.from_pretrained(checkpoint)
```

- 预训练模型的本体只包含基础的 Transformer 模块，对于给定的预处理好的输入，它只会输出一些神经元的值，称为 hidden states 或者特征 (features)。这些 hidden states 通常会被输入到其他的模型部分（称为 head），以完成特定的任务，例如送入到分类头中完成文本分类任务。
![[Pasted image 20241221210657.png]]
- Transformer 模块的输出是一个维度为 (Batch size, Sequence length, Hidden size) 的三维张量，以预处理好的情感分析输入为例
	- Batch size 表示每次输入的样本（文本序列）数量，即每次输入多少个句子，上例中为 2
	- Sequence length 表示文本序列的长度，即每个句子被分为多少个 token，上例中为 16
	- Hidden size 表示每一个 token 经过模型编码后的输出向量（语义表示）的维度。

- 可以打印出这里使用的 distilbert-base 模型的输出维度：
```python
from transformers import AutoTokenizer, AutoModel  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
model = AutoModel.from_pretrained(checkpoint)  
  
raw_inputs = [  
    "I've been waiting for a HuggingFace course my whole life.",  
    "Amazing",  
]  
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  
outputs = model(**inputs)  
print(outputs.last_hidden_state.shape)
```
- Transformers 模型的输出格式类似 字典，可以像上面那样通过属性访问，也可以通过键（`outputs["last_hidden_state"]`），甚至索引访问（`outputs[0]`）。
```txt
torch.Size([2, 16, 768])
```

---

>当使用`**inputs`作为`model`函数的参数时，实际上是在告诉Python：“将`inputs`字典中的所有键值对作为参数传递给`model`函数。”这样做的好处是，不需要显式地为`model`函数的每个参数手动赋值，特别是当`inputs`字典中有很多参数时，这样可以减少代码量并提高代码的可读性。

- 例如，如果`inputs`字典是这样的：
```python
{'input_ids': [1, 2, 3], 'attention_mask': [0, 1, 1], 'token_type_ids': [0, 0, 0]}
```
- 使用`**inputs`后，`model`函数的调用就会变成：
```python
model(input_ids=[1, 2, 3], attention_mask=[0, 1, 1], token_type_ids=[0, 0, 0])
```

---

- 对于情感分析任务，很明显我们最后需要使用的是一个文本分类 head。因此，实际上不会使用 `AutoModel` 类，而是使用 `AutoModelForSequenceClassification`：
```python
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
```
- 对于 batch 中的每一个样本，模型都会输出一个两维的向量（每一维对应一个标签，positive 或 negative），但此时的数值并不适合人类阅读
```txt
tensor([[-1.5607,  1.6123],
        [-4.3321,  4.6592]], grad_fn=<AddmmBackward0>)
torch.Size([2, 2])
```

#### 3.3 后处理
- 模型对第一个句子输出 [−1.5607,1.6123]，对第二个句子输出 [-4.3321,  4.6592]，它们并不是概率值，而是模型最后一层输出的 logits 值。要将他们转换为概率值，还需要让其经过一个 SoftMax 层
```python
import torch  
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)  
print(predictions)
```

```txt
tensor([[4.0195e-02, 9.5980e-01],
        [1.2447e-04, 9.9988e-01]], grad_fn=<SoftmaxBackward0>)
```

>所有 Transformers 模型都会输出 logits 值，因为训练时的损失函数通常会自动结合激活函数（例如 SoftMax）与实际的损失函数（例如交叉熵 cross entropy）。

- 这样就得到了更容易理解的概率值：第一个句子 [0.0402,0.9598]，第二个句子 [0.0124,0.9999]。最后，为了得到对应的标签，可以读取模型 config 中提供的 id2label 属性：
```python
print(model.config.id2label)
```

```txt
{0: 'NEGATIVE', 1: 'POSITIVE'}
```
- 有了这些条件，再对输出稍加处理
```python
predict_tolist = predictions.detach().numpy()  
  
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
  
print(input_list)
```
- 便得到了模型默认的输出格式
```txt
[{'label': 'POSITIVE', 'score': 0.9598}, {'label': 'POSITIVE', 'score': 0.9999}]
```

- 结合以上三个步骤，总结得到pipelines的流程图
![[Pasted image 20241222154456.png]]
### 4. 小结
- 本章内容介绍了如何利用[Hugging Face Model Hub](https://huggingface.co/models)页面，使用Transformers提供的pipelines工具处理各种NLP任务，并探讨了pipelines在后台执行的三个核心步骤：使用Tokenizer对文本进行分词处理；其次，将分词后的数据传递给Model进行处理；最后，将模型的输出转换为人类可读的格式。


## 第二章：模型与分词器

### 1. 模型

- 除了像之前使用 `AutoModel` 根据 checkpoint 自动加载模型以外，也可以直接使用模型对应的 `Model` 类，如 BERT 对应的就是 `BertModel`，但是为了代码的可扩展性，大部分情况还是使用 `AutoModel`。
```python
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

#### 1.1 加载模型

- 所有存储在 [HuggingFace Model Hub](https://huggingface.co/models) 上的模型都可以通过 `Model.from_pretrained()` 来加载权重，也可使用本地路径（预先下载的模型目录）

```python
from transformers import BertModel

model = BertModel.from_pretrained('../models/bert-base-cased')
```

- 该方法会自动缓存下载的模型权重，默认保存在 _~/.cache/huggingface/transformers_，由于 checkpoint 名称加载方式需要连接网络，为了方便可以采用本地路径加载模型。

- 以  `bert-base-cased` 模型的 Hub 页面为例，通常只需要下载模型对应的 _config.json_ 和 _pytorch_model.bin_，以及分词器对应的 _tokenizer.json_、_tokenizer_config.json_ 和 _vocab.txt_。
![[Pasted image 20241222204613.png|475]]


#### 1.2 保存模型

- 保存模型通过调用 `Model.save_pretrained()` 函数实现，例如保存加载的 BERT 模型：
```python
from transformers import AutoModel

model = AutoModel.from_pretrained('bert-base-cased')  
model.save_pretrained('../models/bert-base-cased')
```

- 这会在保存路径下创建两个文件：
	- _config.json_：模型配置文件，存储模型结构参数，例如 Transformer 层数、特征空间维度等；
	- _model.safetensors_：又称为 state dictionary，存储模型的权重。

> 简单来说，配置文件记录模型的**结构**，模型权重记录模型的**参数**，这两个文件缺一不可。

### 2. 分词器

- 神经网络模型不能直接处理文本，因此需要先将文本转换为数字，这个过程被称为**编码 (Encoding)**，包含两个步骤：
	1. 使用分词器 (tokenizer) 将文本按词、子词、字符切分为 tokens；
	2. 将所有的 token 映射到对应的 token ID。

#### 2.1 分词策略

- 根据切分粒度的不同，分词策略可以分为以下几种：

- **按词切分 (Word-based)**
  ![[Pasted image 20241222205322.png]]
    - 文本中所有出现过的独立片段都作为不同的 token，会产生**巨大的词表**。
    - 无法体现出词与词的**关联性**，如，“dog” 和 “dogs”、“run” 和 “running”
    - 当遇到不在词表中的词时，分词器会使用一个专门的 [UNK] token 来表示它是 unknown 的。一个好的分词策略，应该尽可能不出现 unknown token。

> 词表就是一个映射字典，负责将 token 映射到对应的 ID（从 0 开始）。神经网络模型就是通过这些 token ID 来区分每一个 token。

- **按字符切分 (Character-based)**
  ![[Pasted image 20241222205801.png]]
    - 将单词化成字符，只会产生一个非常小的词表，并且很少会出现词表外的 tokens。
    - 但字符本身没有太大意义。
   

> 现在广泛采用的是一种同时结合了按词切分和按字符切分的方式——按子词切分 (Subword tokenization)。

- **按子词切分 (Subword)**
  ![[Pasted image 20241222210313.png]]
	- 高频词直接保留，低频词被切分为更有意义的子词。

#### 2.2 加载与保存分词器

- 分词器的加载与保存与模型相似，例如加载并保存 BERT 模型的分词器：

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')  tokenizer.save_pretrained('../models/bert-base-cased')
```

- 在大部分情况下使用 `AutoTokenizer` 来加载分词器：

```python
from transformers import AutoTokenizer

tokenizer=AutoTokenizer.from_pretrained('bert-base-cased')  
tokenizer.save_pretrained('../models/bert-base-cased')
```

- 调用 `Tokenizer.save_pretrained()` 函数会在保存路径下创建四个文件：
	- _special_tokens_map.json_：映射文件，里面包含 unknown token 等特殊字符的映射关系；
	- _tokenizer_config.json_和_tokenizer.json_：分词器配置文件，存储构建分词器需要的参数；
	- _vocab.txt_：词表，一行一个 token，行号就是对应的 token ID（从 0 开始）。

#### 2.3 编码与解码文本

- 文本编码 (Encoding) 过程包含两个步骤：
	1. **分词**：使用分词器按某种策略将文本切分为 tokens；
	2. **映射**：将 tokens 转化为对应的 token IDs。

- 以 BERT 分词器来对文本进行**分词**：
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

```txt
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
```

>可见，BERT 分词器采用的是子词切分策略，它会不断切分词语直到获得词表中的 token。

- 再将切分出的 tokens 进行**编码**：

```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

```txt
[7993, 170, 13809, 23763, 2443, 1110, 3014]
```

- 使用 `encode()` 将这两个步骤合并，并且 `encode()` 会自动添加模型需要的特殊 token，例如 BERT 分词器会分别在序列的首尾添加 [CLS] 和 [SEP]：

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
sequence_ids = tokenizer.encode(sequence)

print(sequence_ids)
```

```txt
[101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102]
```

> 其中 101 和 102 分别是 [CLS] 和 [SEP] 对应的 token IDs。

---

- 文本解码负责将数字转换成供人类阅读的字符串，也包含两个步骤
	1. 将  token IDs 转化为对应的子词
	2. 根据策略将子词合并
- 使用 `decode()` 解码前面生成的 token IDs
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

decoded_string = tokenizer.decode([101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102])
print(decoded_string)
```

```txt
Using a transformer network is simple
[CLS] Using a Transformer network is simple [SEP]
```

- **在实际编码文本时，最常见的是直接使用分词器进行处理**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_text = tokenizer("Using a Transformer network is simple")
print(tokenized_text)
```
- 这样不仅会返回分词后的 token IDs，**还包含模型需要的其他输入**。
```txt
{
	'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102], 
	 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 
	 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}
```
### 3. 处理多段文本
- 在实际任务中，往往会同时处理多段文本，而模型只接收批 (batch) 数据作为输入，即使只有一段文本，也需要将它组成一个只包含一个样本的 batch，例如：

```python
import torch  
from transformers import AutoModelForSequenceClassification,AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  
sequence = "Using a Transformer network is simple"  

tokens =tokenizer.tokenize(sequence)  
ids =tokenizer.convert_tokens_to_ids(tokens)  

input_ids=torch.tensor([ids])  
print(input_ids)  

outputs = model(input_ids)  
print(outputs)
```

```txt
tensor([[ 2478,  1037, 10938,  2121,  2897,  2003,  3722]])
SequenceClassifierOutput(loss=None, logits=tensor([[ 2.5189, -2.1906]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
```

- 这里通过 `[ids]` 构建了一个只包含一段文本的 batch，更常见的是送入包含多段文本的 batch：

```txt
batched_ids = [ids, ids, ids, ...]
```
---

>上面的代码仅作为演示。**实际场景中，我们应该直接使用分词器对文本进行处理**

- 对于上面的例子：

```python
from transformers import AutoModelForSequenceClassification,AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  
sequence = "Using a Transformer network is simple"

tokens = tokenizer(sequence, return_tensors="pt")  
print(tokens)  

outputs = model(**tokens)  
print(outputs.logits)
```

```txt
{
	'input_ids': tensor([[  101,  2478,  1037, 10938,  2121,  2897,  2003,  3722,   102]]), 
	'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
tensor([[-2.5780,  2.5856]], grad_fn=<AddmmBackward0>)
```
---
#### 3.1 Padding
- 前面展示了如何输入单条文本给模型，现在演示多条文本的情况：
```python
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
outputs=model(**inputs)
```
- 在分词中，添加了新参数 _padding_ 。因为模型要求输入的张量必须是严格的二维矩形，所以需要将分词后长度不一的token序列填充为统一长度，即每一段文本编码后的 token IDs 数量必须一样多。
```txt
{
'input_ids': tensor([
	[  101,  2478,  1037, 10938,  2121,  2897,  2003,  3722,   102],
    [  101,  2651,  2003,  1037,  2204,  2154,   999,   102,     0],
    [  101,  6429,   999,   102,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0]])
}
```

>这种填充是在[SEP] (token IDs =102)字符之后的，即[CLS]和[SEP]标记的是句子的真实开头与结束。

---
- 如果进行手动填充呢？使用 `tokenizer.pad.token_id` 获取当前分词器填充的
```python
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
  
print(model(torch.tensor(sequence1_ids)).logits)  
print(model(torch.tensor(sequence2_ids)).logits)  
print(model(torch.tensor(batched_ids)).logits)
```
- 可见当前分词器采用的padding ID是 `0`，且使用 padding token 填充的序列的结果竟然与其单独送入模型时不同！
```txt
0
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 1.3374, -1.2163]], grad_fn=<AddmmBackward0>)
```
> 因为模型默认会编码输入序列中的所有 token 以建模完整的上下文，即对于手动padding的数据，模型所认为的真正上下文是包含padding在内的。

- 所以，在进行 Padding 操作时，必须明确告知模型哪些 token 是我们填充的，它们不应该参与编码。这就需要使用到 Attention Mask 。

#### 3.2 Attention Mask
- Attention Mask 是一个尺寸与 input IDs 完全相同，且仅由 0 和 1 组成的张量，0 表示对应位置的 token 是填充符，不参与计算。
- 借助Attention Mask 就可标出填充的 padding token 的位置，那该如何将Attention Mask传递给model呢？可以回忆一下 ，在[[How  to use Transforms#Padding]]的第一个例子中，传入model的input中包含什么。
```python 
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
```
- 当然可以通过 `attention_mask` 直接将参数传递给 `model`，此时得到的结果就与模型默认编码的结果一致了。
```txt
tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward0>)
tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
tensor([[ 1.5694, -1.3895],
        [ 0.5803, -0.4125]], grad_fn=<AddmmBackward0>)
```

- 目前大部分 Transformer 模型只能接受长度不超过 512 或 1024 的 token 序列，因此对于长序列，有以下三种处理方法：
	1. 使用一个支持长文的 Transformer 模型，例如 [Longformer](https://huggingface.co/transformers/model_doc/longformer.html) 和 [LED](https://huggingface.co/transformers/model_doc/led.html)（最大长度 4096）；
	2. 设定最大长度 `max_sequence_length` 以**截断**输入序列：`sequence = sequence[:max_sequence_length]`。
	3. 将长文切片为短文本块 (chunk)，然后分别对每一个 chunk 编码。


#### 3.3 直接使用分词器
- 在实际使用中，应该直接使用分词器来完成分词、编码、Padding、构建Attention Mask、截断等操作，下面以 _DistilBERT_ 模型给出一个完整的例子：
```python
from transformers import AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
sequences =["How are you?","Nice to meet you!"]  
  
model_inputs = tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")  
print(model_inputs)
```
- 分词器会给出模型的需要输入，对于 _DistilBERT_ 模型包含 input_ids 和 attention_mask。
```
{'input_ids': tensor([[ 101, 2129, 2024, 2017, 1029,  102,    0],
        [ 101, 3835, 2000, 3113, 2017,  999,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]])}
```

- 下面将具体分析 _padding_、_truncation_、_return_tensors_ 三个参数的作用
---
**Padding 操作**通过 `padding` 参数来控制：
- `padding="longest"`/`padding=True`： 将序列填充到当前 batch 中最长序列的长度；
- `padding="max_length"`：将所有序列填充到模型能够接受的最大长度，例如 BERT 模型就是 512。
```python
from transformers import AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
sequences =["How are you?","Nice to meet you!"]  
  
model_inputs_1 = tokenizer(sequences,padding="longest")  
model_inputs_2 = tokenizer(sequences,padding="max_length")  
print(model_inputs_1)  
print(model_inputs_2)
```

```txt
{'input_ids': [[101, 2129, 2024, 2017, 1029, 102, 0], [101, 3835, 2000, 3113, 2017, 999, 102]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 1]]}

{'input_ids': [[101, 2129, 2024, 2017, 1029, 102, 0, 0, 0,...], [101, 3835, 2000, 3113, 2017, 999, 102, 0, 0, 0,...]], 
'attention_mask': [[1, 1, 1, 1, 1, 1, 0, 0, 0,...], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0,...]]}
```

---
**截断操作**通过 `truncation` 参数来控制：
- `truncation=True`：大于模型最大接受长度的序列都会被截断，例如对于 BERT 模型就会截断长度超过 512 的序列。
- `max_length` ：手动选择控制截断的长度：
```python
from transformers import AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
sequences =["How are you?","Nice to meet you!"]  
  
model_inputs = tokenizer(sequences, max_length=4, truncation=True)  
print(model_inputs)
```

```txt
{'input_ids': [[101, 2129, 2024, 102], [101, 3835, 2000, 102]], 'attention_mask': [[1, 1, 1, 1], [1, 1, 1, 1]]}
```
---
**返回张量类型** 通过`return_tensors` 参数指定返回的张量格式：
-  `pt` ：返回 PyTorch 张量；
- `tf` ：返回 TensorFlow 张量，
- `np` ：返回 NumPy 数组。
```python
from transformers import AutoTokenizer  
  
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
sequences =["How are you?","Nice to meet you!"]  
  
print(tokenizer(sequences,padding=True, return_tensors="pt"))  
print(tokenizer(sequences,padding=True, return_tensors="np"))
```
在设定返回的张量格式之前，要先进行 padding 或截断操作，将batch中的数据处理成统一的长度，才能够将分词的结果送入模型
```txt
{'input_ids': tensor([[ 101, 2129, 2024, 2017, 1029,  102,    0],
        [ 101, 3835, 2000, 3113, 2017,  999,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1]])}
{'input_ids': array([[ 101, 2129, 2024, 2017, 1029,  102,    0],
       [ 101, 3835, 2000, 3113, 2017,  999,  102]]), 'attention_mask': array([[1, 1, 1, 1, 1, 1, 0],
       [1, 1, 1, 1, 1, 1, 1]])}
```


#### 3.4 编码句子对
- 除了对单段文本进行编码以外（batch 只是并行地编码多个单段文本），对于 BERT 等包含句子对（text pair）预训练任务的模型(类似于问答系统)，它们的分词器也支持对句子对进行编码。
- 下面例子使用 _bert-base-uncased_ 模型，对比了使用batch对多段文本进行编码和直接编码句子对的区别：
```python
from transformers import AutoTokenizer  
  
checkpoint = "bert-base-uncased"  
tokenizer = AutoTokenizer.from_pretrained(checkpoint)  
  
sequences = ["Are you Ok?", "I'm OK!"]  
inputs_direct = tokenizer("Are you Ok?", "I'm OK!")  
inputs_indirect = tokenizer(sequences)  
print(inputs_direct)  
print(inputs_indirect)  
print(tokenizer.convert_ids_to_tokens(inputs_direct["input_ids"]))
```

> 在上例中 **Are you Ok?** 和 **I'm OK!** 构成一个句子对

- 通过输出结果可知，句子对的编码思路与单段文本的编码思路并不相同，对于句子对分词器会使用 [SEP] token 拼接两个句子，输出形式为：
$$
[CLS] sentence1 [SEP] sentence2 [SEP]
$$
```txt
{'input_ids': [101, 2024, 2017, 7929, 1029, 102, 1045, 1005, 1049, 7929, 999, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
{'input_ids': [[101, 2024, 2017, 7929, 1029, 102], [101, 1045, 1005, 1049, 7929, 999, 102]], 'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], 'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]}
['[CLS]', 'are', 'you', 'ok', '?', '[SEP]', 'i', "'", 'm', 'ok', '!', '[SEP]']
```
- 对比 _DistilBERT_ 模型，_bert-base-uncased_ 模型的分词器编码结果中，除了input_ids 和 attention_mask，还多了一个关键词 `token_type_ids` ，用于标记哪些 token 属于第一个句子，哪些属于第二个句子，如果将上面例子中的 `token_type_ids` 项与 token 序列对齐
```txt
['[CLS]', 'are', 'you', 'ok', '?', '[SEP]', 'i', "'", 'm', 'ok', '!', '[SEP]']
[   0   ,   0  ,   0  ,   0 ,  0 ,    0   ,  1 ,  1 ,  1 ,   1 ,  1 ,    1   ]
```
- 可见第一个句子 [CLS] sentence1 [SEP] 所有 token 的 type ID 都为 0，而第二个句子sentence2 [SEP]对应的 token type ID 都为 1

> 如果选择其他模型，分词器的输出不一定会包含 `token_type_ids` 项（如 DistilBERT 模型）。分词器只需保证输出格式满足模型所需要的输入格式即可。

-   在实际应用中，可能需要处理大量的句子对，推荐的做法是将它们分别存储在两个数组中。分词器会自动识别这两个数组之间的一一对应关系，并据此构建出相应的句子对。
```python
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
```
- 上例中总共有4组句子对，经过编码得到的 `input_ids` 同样也包含四个数组。
```txt
{'input_ids': tensor([[ 101, 2034, 6251, 1012,  102, 2129, 2024, 2017, 1029,  102],
        [ 101, 2117, 6251, 1012,  102, 1045, 2572, 2986, 1010,  102],
        [ 101, 2353, 2028, 1012,  102, 4067, 2017, 1010,  102,    0],
        [ 101, 2959, 2028, 1012,  102, 1998, 2017, 1029,  102,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])}
torch.Size([4, 10])
```

> 有趣的是，在`token_type_ids`的处理中，对于`sentence2_list`，分词器将句子对的padding部分视为不属于`sentence2`的一部分，可以说与`attention_mask` 有异曲同工之妙。