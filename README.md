  
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