# import and initialize the tokenizer and model from the checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time


def summarizer(chunks):
    checkpoint = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

    str = ""

    start_time = time.time()
    for input in inputs:
        output = model.generate(**input, max_new_tokens = 100)
        str += tokenizer.decode(*output, skip_special_tokens=True)
    end_time = time.time()
    total_time = end_time - start_time

    return str, total_time


