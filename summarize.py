# import and initialize the tokenizer and model from the checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BigBirdPegasusForConditionalGeneration
import time
import torch


device = "cpu"

def summarizer(chunks):
    checkpoint = "sshleifer/distilbart-cnn-12-6"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

    str = ""

    start_time = time.time()
    for input in inputs:
        output = model.generate(**input.to(device), max_new_tokens = 128)
        str += tokenizer.decode(*output.to(device), skip_special_tokens=True)
        str = str.replace("<n>", " ")
    end_time = time.time()
    total_time = end_time - start_time

    return str, total_time



def abstractive_sum(chunks):
    checkpoint = "google/pegasus-arxiv"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)

    inputs = [tokenizer(chunk, return_tensors="pt") for chunk in chunks]

    str = ""

    start_time = time.time()
    for input in inputs:
        output = model.generate(**input.to(device), max_new_tokens = 128)
        str += tokenizer.decode(*output.to(device), skip_special_tokens=True)
        str = str.replace("<n>", " ")
    end_time = time.time()
    total_time = end_time - start_time

    return str, total_time
