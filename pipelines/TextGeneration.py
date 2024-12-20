from transformers import pipeline

generator = pipeline("text-generation")
results = generator("我是你的")
print(results)
results = generator(
    "In this course, we will teach you how to",
    num_return_sequences=2,
    max_length=20
)
print(results)

generator = pipeline("text-generation", model="distilgpt2")
results = generator(
    "In this course, we will teach you how to",
    max_length=20,
    num_return_sequences=1,
)
print(results)


generator = pipeline("text-generation", model="uer/gpt2-chinese-poem")
results = generator(
    "[CLS] 梅 山 如 积 翠 ，",
    max_length=40,
    do_sample=True,
)
print(results)

