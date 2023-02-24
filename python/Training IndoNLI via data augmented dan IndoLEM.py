if __name__ == '__main__':
    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:


    import os.path
    import wget

    if (os.path.exists('./dev.jsonl') == False):
        wget.download("https://huggingface.co/datasets/muhammadravi251001/translated-indo-nli/raw/main/dev.jsonl")
        print(" Selesai download dev.jsonl")
    else: print("File dev.jsonl sudah ada")

    if (os.path.exists('./train.jsonl') == False):
        wget.download("https://huggingface.co/datasets/muhammadravi251001/translated-indo-nli/resolve/main/train.jsonl")
        print(" Selesai download train.jsonl")
    else: print("File train.jsonl sudah ada")

    if (os.path.exists('./dev_augmented.jsonl') == False):
        wget.download("https://huggingface.co/datasets/muhammadravi251001/augmented-indo-nli/raw/main/dev_augmented.jsonl")
        print(" Selesai download dev_augmented.jsonl")
    else: print("File dev_augmented.jsonl sudah ada")

    if (os.path.exists('./train_augmented.jsonl') == False):
        wget.download("https://huggingface.co/datasets/muhammadravi251001/augmented-indo-nli/resolve/main/train_augmented.jsonl")
        print(" Selesai download train_augmented.jsonl")
    else: print("File train_augmented.jsonl sudah ada")

    # ## Mendefinisikan hyperparameter

    # In[2]:
    import sys

    MODEL_NAME = "indolem/indobert-base-uncased"
    # EPOCH = 1
    # SAMPLE = 25
    EPOCH = 16
    SAMPLE = sys.maxsize

    SEED = 42
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION = 4
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 400
    STRIDE = 100
    LOGGING_STEPS = 50
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01


    # ## Instalasi setiap module yang digunakan

    # In[3]:


    #get_ipython().system('pip install -r requirements.txt')


    # In[4]:


    #get_ipython().system('nvidia-smi')


    # In[5]:


    import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


    # ## Import setiap library yang digunakan

    # In[6]:


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
    Dataset,
    DatasetDict
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


    # In[7]:


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ## Gunakan tokenizer yang sudah pre-trained

    # In[8]:


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


    # ## Import dataset IndoNLI

    # In[9]:


    data_train = pd.read_json(path_or_buf='train_augmented.jsonl', lines=True)
    data_train


    # In[10]:


    data_validation = pd.read_json(path_or_buf='dev_augmented.jsonl', lines=True)
    data_validation


    # In[11]:


    data_train = data_train[data_train.label != '-']
    data_validation = data_validation[data_validation.label != '-']

    train_dataset = Dataset.from_dict(data_train)
    validation_dataset = Dataset.from_dict(data_validation)

    data_indonli_augmented = DatasetDict({"train": train_dataset, "validation": validation_dataset})


    # In[12]:


    data_indonli = data_indonli_augmented


    # ## Fungsi utilitas untuk pre-process data IndoNLI

    # In[13]:


    def preprocess_function_indonli(examples, tokenizer, MAX_LENGTH):
        return tokenizer(
            examples['premise'], examples['hypothesis'],
            truncation=True, return_token_type_ids=True,
            max_length=MAX_LENGTH
        )


    # ## Melakukan tokenisasi data IndoNLI

    # In[14]:


    tokenized_data_indonli = data_indonli.map(
        preprocess_function_indonli,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=['premise', 'hypothesis'],
        fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH}
    )


    # In[15]:


    tokenized_data_indonli.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)


    # In[16]:


    tokenized_data_indonli_train = Dataset.from_dict(tokenized_data_indonli["train"][:SAMPLE])
    tokenized_data_indonli_validation = Dataset.from_dict(tokenized_data_indonli["validation"][:SAMPLE])


    # # Tahapan fine-tune IndoNLI diatas IndoBERT

    # ## Fungsi utilitas untuk komputasi metrik

    # In[17]:


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(
            predictions=predictions, references=labels)


    # ## Dictionary untuk mapping label

    # In[18]:


    id2label = {0: 'entailment', 1: 'neutral', 
                2: 'contradiction'}
    label2id = {'entailment': 0, 'neutral': 
                1, 'contradiction': 2}
    accuracy = evaluate.load('accuracy')


    # ## Gunakan model Sequence Classification yang sudah pre-trained

    # In[19]:


    model_sc = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, 
        id2label=id2label, label2id=label2id)


    # In[20]:


    model_sc = model_sc.to(device)


    # ## Melakukan pengumpulan data dengan padding

    # In[21]:


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # ## Mendefinisikan argumen (dataops) untuk training nanti

    # In[22]:


    TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
    NAME = 'IndoNLI-data_augmented-with_IndoLEM'
    SC = f'./results/{NAME}-{TIME_NOW}'

    CHECKPOINT_DIR = f'{SC}/checkpoint/'
    MODEL_DIR = f'{SC}/model/'
    OUTPUT_DIR = f'{SC}/output/'
    ACCURACY_DIR = f'{SC}/accuracy/'

    REPO_NAME = f'fine-tuned-{NAME}'


    # In[23]:


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

    # In[24]:


    trainer_sc = Trainer(
        model=model_sc,
        args=training_args_sc,
        train_dataset=tokenized_data_indonli_train,
        eval_dataset=tokenized_data_indonli_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    # In[25]:


    trainer_sc.train()


    # ## Simpan model Sequence Classification

    # In[26]:


    trainer_sc.save_model(MODEL_DIR)


    # # Melakukan prediksi dari model

    # In[27]:


    predict_result = trainer_sc.predict(tokenized_data_indonli_validation)


    # In[28]:


    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()


    # # Melakukan evaluasi dari prediksi

    # In[29]:


    def compute_accuracy(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(
            predictions=predictions, references=labels)


    # In[30]:


    accuracy_result = compute_accuracy(predict_result)


    # In[31]:


    os.makedirs(os.path.dirname(ACCURACY_DIR), exist_ok=True)
    with open(f'{ACCURACY_DIR}/accuracy.txt', "w") as f:
        f.write(str(accuracy_result))
        f.close()

