from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


text = ""
with open("example.txt") as f:
    for line in f:
        text += line

print(summarizer(text, max_length=130, min_length=30, do_sample=False))
