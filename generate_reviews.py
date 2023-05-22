import gc

import numpy as np
import pandas as pd
import torch
from pynvml import *
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer)


def generate(text, **kwargs):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt',
                       truncation=True).to(model.device)
    with torch.no_grad():
        hypotheses = model.generate(
            **inputs, num_beams=5, max_length=30, min_length=15, **kwargs)
    return tokenizer.decode(hypotheses[0], skip_special_tokens=True)


df = pd.read_csv('data/wildberries-comments-promts/data_with_promts')
df.loc[df.index, 'name'] = df['mark_token'] + df['name']
df = df.sample(119993, random_state=1337)

val_df = df.sample(40, random_state=1337)
val_idx = val_df.index
df.drop(index=val_idx, inplace=True)
train_df = df

train_x = train_df['name'].to_numpy()
train_y = train_df['text'].to_numpy()
val_x = val_df['name'].to_numpy()
val_y = val_df['text'].to_numpy()


tokenizer = AutoTokenizer.from_pretrained("results/", local_files_only=True)
tokenizer.src_lang = 'ru'

model = AutoModelForSeq2SeqLM.from_pretrained(
    "results/", local_files_only=True)
model.to('cuda:0')

description = 'ужасный отзыв Кольца'
print(generate(description))

for i, row in val_df.iterrows():
    print(row['name'])
    print(generate(row['name']))
    print('=====================================')
