#!/usr/bin/env python
# coding: utf-8

# # Menjalankan QA Tanpa Intermediate Task - Transfer Learning

# # Import semua module

# In[19]:


#!pip install datasets
#!pip install transformers
#!pip install tensorboard
#!pip install evaluate
#!pip install git+https://github.com/IndoNLP/nusa-crowd.git@release_exp


# In[20]:


get_ipython().system('pip install -r requirements.txt')


# In[21]:


# Melihat GPU yang tersedia dan penggunaannya.
get_ipython().system('nvidia-smi')


# In[22]:


# Memilih GPU yang akan digunakan (contohnya: GPU #7)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# In[54]:


import transformers
import evaluate
import torch
import operator
import ast
import json
import re
import sys

import numpy as np
import pandas as pd
import torch.nn as nn

from multiprocessing import cpu_count
from evaluate import load
from nusacrowd import NusantaraConfigHelper
from torch.utils.data import DataLoader
from datetime import datetime
from huggingface_hub import notebook_login

from datasets import (
    load_dataset, 
    load_from_disk,
    Dataset
)
from transformers import (
    BigBirdTokenizerFast,
    BigBirdForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    BertForSequenceClassification,
    BertForQuestionAnswering,
    AutoModel, 
    BertTokenizerFast,
    AutoTokenizer, 
    AutoModel, 
    BertTokenizer, 
    BertForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering
)


# # Definisikan hyperparameter

# In[24]:


MODEL_NAME = "indolem/indobert-base-uncased"
SEED = 42
EPOCH = 1
BATCH_SIZE = 16
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 1e-5
MAX_LENGTH = 400
STRIDE = 100
LOGGING_STEPS = 50
WARMUP_RATIO = 0.06
WEIGHT_DECAY = 0.01
# Untuk mempercepat training, saya ubah SAMPLE menjadi 100.
# Bila mau menggunakan keseluruhan data, gunakan: 
# SAMPLE = sys.maxsize
SAMPLE = 10


# # Import data SQUAD-ID

# In[25]:


conhelps = NusantaraConfigHelper()
data_squad_id = conhelps.filtered(lambda x: 'squad_id' in x.dataset_name)[0].load_dataset()
data_squad_id


# In[26]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# # Definisikan tokenizer

# In[27]:


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# # Definisikan fungsi pre-processnya

# In[28]:


def rindex(lst, value, operator=operator):
      return len(lst) - operator.indexOf(reversed(lst), value) - 1

def preprocess_function_qa(examples, tokenizer, MAX_LENGTH=MAX_LENGTH, STRIDE=STRIDE, rindex=rindex, operator=operator):
    examples["question"] = [q.lstrip() for q in examples["question"]]
    examples["context"] = [c.lstrip() for c in examples["context"]]

    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        max_length = MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors='np'
    )

    tokenized_examples['start_positions'] = []
    tokenized_examples['end_positions'] = []

    for seq_idx in range(len(tokenized_examples['input_ids'])):
      seq_ids = tokenized_examples.sequence_ids(seq_idx)
      offset_mappings = tokenized_examples['offset_mapping'][seq_idx]

      cur_example_idx = tokenized_examples['overflow_to_sample_mapping'][seq_idx]

      #answer = examples['answer'][seq_idx][0]
      answer = examples['answer'][cur_example_idx][0]
      answer = eval(answer)

      #answer_text = answer['text'][0]
      answer_start = answer['answer_start']
      #answer_end = answer_start + len(answer_text)
      answer_end = answer['answer_end']

      context_pos_start = seq_ids.index(1)
      context_pos_end = rindex(seq_ids, 1, operator)

      s = e = 0
      if (offset_mappings[context_pos_start][0] <= answer_start and
          offset_mappings[context_pos_end][1] >= answer_end):
        i = context_pos_start
        while offset_mappings[i][0] < answer_start:
          i += 1
        if offset_mappings[i][0] == answer_start:
          s = i
        else:
          s = i - 1

        j = context_pos_end
        while offset_mappings[j][1] > answer_end:
          j -= 1      
        if offset_mappings[j][1] == answer_end:
          e = j
        else:
          e = j + 1

      tokenized_examples['start_positions'].append(s)
      tokenized_examples['end_positions'].append(e)
    return tokenized_examples


# # Mulai tokenisasi dan pre-process

# In[29]:


tokenized_data_squad_id = data_squad_id.map(
    preprocess_function_qa,
    batched=True,
    remove_columns=data_squad_id["train"].column_names,
    num_proc=2,
    #fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH, 'STRIDE': STRIDE, 'rindex': rindex, 'operator': operator}
    fn_kwargs={'tokenizer': tokenizer}
)


# In[30]:


tokenized_data_squad_id = tokenized_data_squad_id.remove_columns(["offset_mapping", 
                                          "overflow_to_sample_mapping"])


# In[31]:


tokenized_data_squad_id


# In[32]:


tokenized_data_squad_id.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)


# In[34]:


tokenized_data_squad_train = Dataset.from_dict(tokenized_data_squad_id["train"][:SAMPLE])
tokenized_data_squad_validation = Dataset.from_dict(tokenized_data_squad_id["validation"][:SAMPLE])


# # Mendefinisikan argumen (dataops) untuk training nanti

# In[97]:


TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
QA = './results/mBERT-without-intermediate'
CHECKPOINT_DIR = f'{QA}-{TIME_NOW}/checkpoint/'
MODEL_DIR = f'{QA}-{TIME_NOW}/model/'
OUTPUT_DIR = f'{QA}-{TIME_NOW}/output/'
ACCURACY_DIR = f'{QA}-{TIME_NOW}/accuracy/'


# # Mendefinisikan Training Arguments untuk train

# In[98]:


training_args_qa = TrainingArguments(
    
    # Checkpoint
    output_dir=CHECKPOINT_DIR,
    save_strategy='epoch',
    save_total_limit=EPOCH,
    
    # Log
    report_to='tensorboard',
    logging_strategy='steps',
    logging_first_step=True,
    logging_steps=LOGGING_STEPS,
    
    # Train
    num_train_epochs=EPOCH,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    bf16=False,
    dataloader_num_workers=cpu_count(),
    
    # Miscellaneous
    evaluation_strategy='epoch',
    seed=SEED,
)


# # Pendefinisian model Question Answering

# In[99]:


model_qa = BertForQuestionAnswering.from_pretrained(MODEL_NAME)


# In[100]:


model_qa = model_qa.to(device)


# # Melakukan pengumpulan data dengan padding

# In[101]:


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# # Mulai training untuk fine-tune SQUAD diatas IndoBERT

# In[102]:


trainer_qa = Trainer(
    model=model_qa.to(device),
    args=training_args_qa,
    train_dataset=tokenized_data_squad_train,
    eval_dataset=tokenized_data_squad_validation,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# In[103]:


trainer_qa.train()


# # Menyimpan model Question Answering

# In[104]:


trainer_qa.save_model(MODEL_DIR)


# # Melakukan prediksi dari model

# In[105]:


predict_result = trainer_qa.predict(tokenized_data_squad_validation)


# In[106]:


os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
  f.write(str(predict_result))
  f.close()


# # Melakukan evaluasi dari prediksi

# In[107]:


def compute_accuracy(predict_result):
    predictions_idx = np.argmax(
        predict_result.predictions, axis=2)
    total_correct = 0
    denominator = len(predictions_idx[0])
    label_array = np.asarray(predict_result.label_ids)

    for i in range(len(predict_result.predictions[0])):
      if predictions_idx[0][i] == label_array[0][i]:
        if predictions_idx[1][i] == label_array[1][i]:
          total_correct += 1

    accuracy = (total_correct / denominator)
    return accuracy


# In[108]:


accuracy_result = compute_accuracy(predict_result)


# In[109]:


os.makedirs(os.path.dirname(ACCURACY_DIR), exist_ok=True)
with open(f'{ACCURACY_DIR}/accuracy.txt', "w") as f:
  f.write(str(accuracy_result))
  f.close()


# ## Push Trainer ke HuggingFace

# In[110]:


notebook_login()


# In[111]:


get_ipython().system('pip install git-lfs')


# In[115]:


trainer_qa.push_to_hub()

