from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch


model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = "summarize: "
with open("example.txt") as f:
    for line in f:
        text += line

# print(text)
input_ids = tokenizer([text], max_length=1028, truncation=True, return_tensors="pt").input_ids
with torch.no_grad():
    generated_ids = model.generate(input_ids=input_ids, num_beams=2, max_new_tokens=128,  repetition_penalty=2.5, length_penalty=1.0, early_stopping=True)
    summarization = tokenizer.decode(generated_ids[0], skip_special_tokens=True)


print(summarization)
