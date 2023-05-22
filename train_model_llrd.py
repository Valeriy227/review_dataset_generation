import gc

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from pynvml import *
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          Trainer, TrainingArguments, DataCollatorForSeq2Seq, AdamW, get_cosine_schedule_with_warmup)


def get_optimizer_grouped_parameters(
    model, model_type,
    learning_rate, weight_decay,
    layerwise_learning_rate_decay
):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": 0.0,
            "lr": learning_rate,
        },
    ]
    layers = [model.shared] + list(model.encoder.block)

    layers.reverse()
    lr = learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


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
df = pd.read_csv('/kaggle/input/wildberries-comments-promts/data_with_promts')
df.loc[df.index, 'name'] = df['gender_token'] + df['mark_token'] + df['name']

train_idx, val_idx = [], []

train_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                  & (df['mark_token'] == 'ужасный отзыв ')][:830].index)
train_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                  & (df['mark_token'] == 'плохой отзыв ')][:480].index)
train_idx += list(df[(df['gender_token'] == 'мужской отзыв ') & (df['mark_token']
                  == 'отличный отзыв ')].sample(2000, random_state=228)[:1995].index)
train_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                  & (df['mark_token'] == 'нормальный отзыв ')][:745].index)
train_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                  == 'ужасный отзыв ')].sample(2000, random_state=228)[:1995].index)
train_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                  == 'плохой отзыв ')].sample(2000, random_state=228)[:1995].index)
train_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                  == 'отличный отзыв ')].sample(2000, random_state=228)[:1995].index)
train_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                  == 'нормальный отзыв ')].sample(2000, random_state=228)[:1995].index)

val_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                & (df['mark_token'] == 'ужасный отзыв ')][830:].index)
val_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                & (df['mark_token'] == 'плохой отзыв ')][480:].index)
val_idx += list(df[(df['gender_token'] == 'мужской отзыв ') & (df['mark_token']
                == 'отличный отзыв ')].sample(2000, random_state=228)[1995:].index)
val_idx += list(df[(df['gender_token'] == 'мужской отзыв ')
                & (df['mark_token'] == 'нормальный отзыв ')][745:].index)
val_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                == 'ужасный отзыв ')].sample(2000, random_state=228)[1995:].index)
val_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                == 'плохой отзыв ')].sample(2000, random_state=228)[1995:].index)
val_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                == 'отличный отзыв ')].sample(2000, random_state=228)[1995:].index)
val_idx += list(df[(df['gender_token'] == 'женский отзыв ') & (df['mark_token']
                == 'нормальный отзыв ')].sample(2000, random_state=228)[1995:].index)

train_df = df.loc[train_idx]
train_df.reset_index()
val_df = df.loc[val_idx]
val_df.reset_index()

train_x = train_df['description'].to_numpy()
train_y = train_df['text'].to_numpy()
val_x = val_df['description'].to_numpy()
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
    run_name='T5_llrd',
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=6,
    weight_decay=0.01,
    save_strategy="no",
    report_to='wandb'
)


layerwise_learning_rate_decay = 0.9

grouped_optimizer_params = get_optimizer_grouped_parameters(
    model,
    None,
    training_args.learning_rate,
    training_args.weight_decay,
    layerwise_learning_rate_decay
)

optimizer = AdamW(
    grouped_optimizer_params,
    lr=training_args.learning_rate,
    eps=training_args.adam_epsilon,
    correct_bias=not False
)

num_training_steps = training_args.num_train_epochs * len(ds_train)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
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
    compute_metrics=compute_cosin,
    optimizers=(optimizer, scheduler)
)
trainer.train()
trainer.save_model()
wandb.finish()
