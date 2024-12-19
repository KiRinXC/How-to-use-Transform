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