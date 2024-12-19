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