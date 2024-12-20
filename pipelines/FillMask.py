from transformers import pipeline

unmasker = pipeline("fill-mask")
results = unmasker("This course <mask> will teach you all about <mask> models.",
                   top_k=1)
print(results)