  
## 一、项目启动  
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
- 使用的模型是 _gpt2_，显然该没有生成中文文本的能力
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
- 有过深度学习基础项目经验的同学应该能够很简单的理解pipelines。pipelines本质上是一个端到端的系统，我们只在乎其输入和输出，而不管其中间做了什么。以手写字识别为例，我们只关注输入的图像最终变成了哪个数字，而不关心其中间过程。
- 但为了弄懂这一切为何如此，还是需要剥开pipelines的外表，细究其内在运行机理。