from transformers import pipeline
question_answerer = pipeline("question-answering", model='distilbert-base-cased-distilled-squad')

context = r"My name is Sylvain and I work at Hugging Face in Brooklyn"

result = question_answerer(
    question="Where do I work?",
    context=context)
print(result)
