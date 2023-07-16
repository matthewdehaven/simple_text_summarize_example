from gensim.summarization.summarizer import summarize

with open("example.txt") as f:
    text = f.read()

print(summarize(text))
