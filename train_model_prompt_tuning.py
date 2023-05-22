import gc

import evaluate
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset
from peft import (PromptTuningConfig, get_peft_model)
from pynvml import *
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          Trainer, TrainingArguments, DefaultDataCollator)


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


peft_config = PromptTuningConfig(
    peft_type="PROMPT_TUNING",
    task_type="SEQ_2_SEQ_LM",
    inference_mode=False,
    num_virtual_tokens=20,
    num_transformer_submodules=2,
    num_attention_heads=12,
    num_layers=12,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Напиши отзыв на этот товар",
    tokenizer_name_or_path="/kaggle/working/rut5-base-multitask",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# tqdm.pandas(desc="my bar!")

ds_train = Dataset.from_dict({"description": train_x, "review": train_y})
ds_val = Dataset.from_dict({"description": val_x, "review": val_y})

tokenized_x = ds_train.map(tokenizer_function, batched=True, remove_columns=[
                           'description', 'review'])
tokenized_y = ds_val.map(tokenizer_function, batched=True,
                         remove_columns=['description', 'review'])

data_collator = DefaultDataCollator()

gc.collect()
torch.cuda.empty_cache()

training_args = TrainingArguments(
    run_name='T5_prompt_tuning',
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=8,
    weight_decay=0.01,
    save_strategy="no",
    eval_accumulation_steps=2,
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
