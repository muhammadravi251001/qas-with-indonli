if __name__ == '__main__':  
  #!/usr/bin/env python
  # coding: utf-8

  # ## Instalasi setiap module yang digunakan

  # In[1]:


  #get_ipython().system('pip install -r requirements.txt')


  # In[2]:


  #get_ipython().system('nvidia-smi')


  # In[3]:


  import os
  #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


  # ## Import setiap library yang digunakan

  # In[4]:


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


  # ## Mendefinisikan hyperparameter

  # In[5]:


  MODEL_NAME = "xlm-roberta-base"
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
  SAMPLE = sys.maxsize
  # SAMPLE = 100


  # In[6]:


  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


  # ## Gunakan tokenizer yang sudah pre-trained

  # In[7]:


  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


  # ## Import dataset IndoNLI

  # In[8]:


  data_indonli = load_dataset("indonli")


  # ## Fungsi utilitas untuk pre-process data IndoNLI

  # In[9]:


  def preprocess_function_indonli(examples, tokenizer):
      return tokenizer(
          examples['premise'], examples['hypothesis'],
          truncation=True, return_token_type_ids=True
      )


  # ## Melakukan tokenisasi data IndoNLI

  # In[10]:


  tokenized_data_indonli = data_indonli.map(
      preprocess_function_indonli,
      batched=True,
      load_from_cache_file=True,
      num_proc=1,
      remove_columns=['premise', 'hypothesis'],
      fn_kwargs={'tokenizer': tokenizer}
  )


  # In[11]:


  tokenized_data_indonli.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)


  # In[12]:


  tokenized_data_indonli_train = Dataset.from_dict(tokenized_data_indonli["train"][:SAMPLE])
  tokenized_data_indonli_validation = Dataset.from_dict(tokenized_data_indonli["validation"][:SAMPLE])
  tokenized_data_indonli_test_lay = Dataset.from_dict(tokenized_data_indonli["test_lay"][:SAMPLE])
  tokenized_data_indonli_test_expert = Dataset.from_dict(tokenized_data_indonli["test_expert"][:SAMPLE])


  # # Tahapan fine-tune IndoNLI diatas IndoBERT

  # ## Fungsi utilitas untuk komputasi metrik

  # In[13]:


  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = np.argmax(predictions, axis=1)
      return accuracy.compute(
          predictions=predictions, references=labels)


  # ## Dictionary untuk mapping label

  # In[14]:


  id2label = {0: 'entailment', 1: 'neutral', 
              2: 'contradiction'}
  label2id = {'entailment': 0, 'neutral': 
              1, 'contradiction': 2}
  accuracy = evaluate.load('accuracy')


  # ## Gunakan model Sequence Classification yang sudah pre-trained

  # In[15]:


  model_sc = BertForSequenceClassification.from_pretrained(
      MODEL_NAME, num_labels=3, 
      id2label=id2label, label2id=label2id)


  # In[16]:


  model_sc = model_sc.to(device)


  # ## Melakukan pengumpulan data dengan padding

  # In[17]:


  data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


  # ## Mendefinisikan argumen (dataops) untuk training nanti

  # In[18]:


  TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
  NAME = 'IndoNLI-data_train-with_XLMR'
  SC = f'./results/{NAME}/'
  CHECKPOINT_DIR = f'{SC}-{TIME_NOW}/checkpoint/'
  MODEL_DIR = f'{SC}-{TIME_NOW}/model/'
  OUTPUT_DIR = f'{SC}-{TIME_NOW}/output/'
  ACCURACY_DIR = f'{SC}-{TIME_NOW}/accuracy/'
  REPO_NAME = f'fine-tuned-{NAME}'


  # In[19]:


  training_args_sc = TrainingArguments(
      
      # Checkpoint
      output_dir=CHECKPOINT_DIR,
      overwrite_output_dir=True,
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
      push_to_hub=True,
      hub_model_id=REPO_NAME
  )


  # ## Mulai training untuk fine-tune IndoNLI diatas IndoBERT

  # In[20]:


  trainer_sc = Trainer(
      model=model_sc,
      args=training_args_sc,
      train_dataset=tokenized_data_indonli_train,
      eval_dataset=tokenized_data_indonli_validation,
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )


  # In[21]:


  trainer_sc.train()


  # ## Simpan model Sequence Classification

  # In[22]:


  trainer_sc.save_model(MODEL_DIR)


  # # Melakukan prediksi dari model

  # In[23]:


  predict_result = trainer_sc.predict(tokenized_data_indonli_validation)


  # In[24]:


  os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
  with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
    f.write(str(predict_result))
    f.close()


  # # Melakukan evaluasi dari prediksi

  # In[25]:


  def compute_accuracy(eval_pred):
      predictions = eval_pred.predictions
      labels = eval_pred.label_ids
      predictions = np.argmax(predictions, axis=1)
      return accuracy.compute(
          predictions=predictions, references=labels)


  # In[26]:


  accuracy_result = compute_accuracy(predict_result)


  # In[27]:


  os.makedirs(os.path.dirname(ACCURACY_DIR), exist_ok=True)
  with open(f'{ACCURACY_DIR}/accuracy.txt', "w") as f:
    f.write(str(accuracy_result))
    f.close()


  # ## Push Trainer ke HuggingFace

  # In[28]:


  #notebook_login()


  # In[29]:


  #!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
  #!apt-get install git-lfs


  # In[30]:


  #REPO_NAME = 'muhammadravi251001/finetuned-SC-indobert-on-indonli_basic-train'
  #trainer_sc.push_to_hub(repo_id=REPO_NAME)

