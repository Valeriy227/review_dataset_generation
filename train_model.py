import gc

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from pynvml import *
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq,
                          Trainer, TrainingArguments)


def compute_cosin(eval_data):
    gen_set, valid_set = eval_data
    predictions = np.argmax(gen_set[0], axis=-1)
    gen_set_decode = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    valid_set_decode = tokenizer.batch_decode(
        valid_set, skip_special_tokens=True)
    results = bertscore.compute(
        predictions=gen_set_decode, references=valid_set_decode, model_type="distilbert-base-uncased")
    results['precision'] = np.mean(results['precision'])
    results['recall'] = np.mean(results['recall'])
    results['f1'] = np.mean(results['f1'])
    return results


def tokenizer_function(texts):
    return tokenizer(text=texts["description"], text_target=texts["review"], padding=True, truncation=True)


# Предобработка данных
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

# Модель

tokenizer = AutoTokenizer.from_pretrained(
    "rut5-base-multitask/", local_files_only=True)
tokenizer.src_lang = 'ru'

model = AutoModelForSeq2SeqLM.from_pretrained(
    "rut5-base-multitask/", local_files_only=True)

# tqdm.pandas(desc="my bar!")

ds_train = Dataset.from_dict({"description": train_x, "review": train_y})
ds_val = Dataset.from_dict({"description": val_x, "review": val_y})

tokenized_x = ds_train.map(tokenizer_function, batched=True, remove_columns=[
                           'description', 'review'])
tokenized_y = ds_val.map(tokenizer_function, batched=True,
                         remove_columns=['description', 'review'])

data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer)

gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(
    run_name='T5_names_const_lr_large_1',
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="no",
    lr_scheduler_type='constant_with_warmup'
)

device = 'cuda:0'
bertscore = evaluate.load("bertscore")

wandb.login(key='---')
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_x,
    eval_dataset=tokenized_y,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_cosin
)
trainer.train()
trainer.save_model()
wandb.finish()
