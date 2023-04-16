import argparse
import sys

parser = argparse.ArgumentParser(description="Program untuk training IndoNLI")
parser.add_argument('-m', '--model_name', type=str, metavar='', required=True, help="Nama model Anda; String; choice=[indolem, indonlu, xlmr, your model choice]")
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True, help="Nama dataset Anda; String; choice=[basic, translated, augmented]")
parser.add_argument('-e', '--epoch', type=int, metavar='', required=True, help="Jumlah epoch Anda; Integer; choice=[all integer]")
parser.add_argument('-sa', '--sample', type=str, metavar='', required=True, help="Jumlah sampling data Anda; Integer; choice=[max, all integer]")
parser.add_argument('-l', '--learn_rate', type=str, metavar='', required=False, help="Jumlah learning rate Anda; Float; choice=[all float]; default=1e-5", default=1e-5)
parser.add_argument('-se', '--seed', type=int, metavar='', required=False, help="Jumlah seed Anda; Integer; choice=[all integer]; default=42", default=42)
parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=False, help="Jumlah batch-size Anda; Integer; choice=[all integer]; default=32", default=32)
parser.add_argument('-t', '--token', type=str, metavar='', required=False, help="Token Hugging Face Anda; String; choice=[all string token]; default=(TOKEN_HF_muhammadravi251001)", default="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")
args = parser.parse_args()

if __name__ == "__main__":
    
    if (args.model_name) == "indolem":
        MODEL_NAME = "indolem/indobert-base-uncased"
    elif (args.model_name) == "indonlu":
        MODEL_NAME = "indobenchmark/indobert-large-p2"
    elif (args.model_name) == "xlmr":
        MODEL_NAME = "xlm-roberta-large"
    else: MODEL_NAME = str(args.model_name)
    
    if (args.data_name) == "basic":
        DATA_NAME = "Basic"
    elif (args.data_name) == "translated":
        DATA_NAME = "Translated"
    elif (args.data_name) == "augmented":
        DATA_NAME = "Augmented"

    if (args.sample) == "max":
        SAMPLE = sys.maxsize
    else: SAMPLE = int(args.sample)

    EPOCH = int(args.epoch)
    LEARNING_RATE = float(args.learn_rate)
    SEED = int(args.seed)
    HUB_TOKEN = str(args.token)
    BATCH_SIZE = int(args.batch_size)

    if HUB_TOKEN == "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": USER = "muhammadravi251001"
    else: USER = "afaji"

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

    print("Program training IndoNLI mulai...")
    print(f"Mulai training IndoNLI dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, dan token: {HUB_TOKEN}")

    # ## Mendefinisikan hyperparameter
    MODEL_NAME = MODEL_NAME
    EPOCH = EPOCH
    SAMPLE = SAMPLE
    LEARNING_RATE = LEARNING_RATE
    HUB_TOKEN = HUB_TOKEN
    SEED = SEED
    BATCH_SIZE = BATCH_SIZE
    
    GRADIENT_ACCUMULATION = 8
    MAX_LENGTH = 512
    STRIDE = 128
    LOGGING_STEPS = 50
    WARMUP_RATIO = 0.0
    WEIGHT_DECAY = 0.0
    EVAL_STEPS_RATIO = 0.5

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

    # ## Import setiap library yang digunakan
    import evaluate
    import torch
    import re

    import numpy as np
    import pandas as pd

    from multiprocessing import cpu_count
    from evaluate import load
    from datetime import datetime
    from huggingface_hub import notebook_login
    from tqdm import tqdm
    from huggingface_hub import HfApi

    from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    )
    from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback, 
    AutoModel
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ## Gunakan tokenizer yang sudah pre-trained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ## Import dataset IndoNLI
    if (DATA_NAME == "Basic"):
        data_indonli = load_dataset("indonli")

    elif (DATA_NAME == "Translated"):
        data_train = pd.read_json(path_or_buf='train.jsonl', lines=True)
        data_train = data_train[['sentence1', 'sentence2', 'gold_label']]
        data_train = data_train.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
        data_train['label'] = data_train['label'].replace(['entailment'], 0)
        data_train['label'] = data_train['label'].replace(['contradiction'], 1)
        data_train['label'] = data_train['label'].replace(['neutral'], 2)

        data_validation = pd.read_json(path_or_buf='dev.jsonl', lines=True)
        data_validation = data_validation[['sentence1', 'sentence2', 'gold_label']]
        data_validation = data_validation.rename(columns={'sentence1': 'premise', 'sentence2': 'hypothesis', 'gold_label': 'label'})
        data_validation['label'] = data_validation['label'].replace(['entailment'], 0)
        data_validation['label'] = data_validation['label'].replace(['contradiction'], 1)
        data_validation['label'] = data_validation['label'].replace(['neutral'], 2)

        data_train = data_train[data_train.label != '-']
        data_validation = data_validation[data_validation.label != '-']
        train_dataset = Dataset.from_dict(data_train)
        validation_dataset = Dataset.from_dict(data_validation)

        data_indonli_translated = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        data_indonli = data_indonli_translated

    elif (DATA_NAME == "Augmented"):
        data_train = pd.read_json(path_or_buf='train_augmented.jsonl', lines=True)
        data_validation = pd.read_json(path_or_buf='dev_augmented.jsonl', lines=True)

        data_train = data_train[data_train.label != '-']
        data_validation = data_validation[data_validation.label != '-']

        train_dataset = Dataset.from_dict(data_train)
        validation_dataset = Dataset.from_dict(data_validation)

        data_indonli_augmented = DatasetDict({"train": train_dataset, "validation": validation_dataset})
        data_indonli = data_indonli_augmented

    # ## Fungsi utilitas untuk pre-process data IndoNLI
    def preprocess_function_indonli(examples, tokenizer, MAX_LENGTH):
        return tokenizer(
            examples['premise'], examples['hypothesis'],
            truncation=True, return_token_type_ids=True,
            max_length=MAX_LENGTH
        )

    # ## Melakukan tokenisasi data IndoNLI
    tokenized_data_indonli = data_indonli.map(
        preprocess_function_indonli,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=['premise', 'hypothesis'],
        fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH}
    )

    tokenized_data_indonli.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)
    tokenized_data_indonli_train = Dataset.from_dict(tokenized_data_indonli["train"][:SAMPLE])
    tokenized_data_indonli_validation = Dataset.from_dict(tokenized_data_indonli["validation"][:SAMPLE])

    # ## Dictionary untuk mapping label
    id2label = {0: 'entailment', 1: 'neutral', 
                2: 'contradiction'}
    label2id = {'entailment': 0, 'neutral': 
                1, 'contradiction': 2}
    
    # ## Menggunakan library evaluate untuk evaluasi metrik
    accuracy = evaluate.load('accuracy')
    f1 = evaluate.load('f1')

    # # Tahapan fine-tune IndoNLI diatas IndoBERT
    # ## Fungsi utilitas untuk komputasi metrik
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(predictions, axis=1)
        
        acc_result = accuracy.compute(predictions=predictions, references=labels)
        f1_result = f1.compute(predictions=predictions, references=labels, average="weighted")

        return {'accuracy': acc_result['accuracy'], 'f1': f1_result['f1']}

    # ## Gunakan model Sequence Classification yang sudah pre-trained
    model_sc = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, 
        id2label=id2label, label2id=label2id)

    model_sc = model_sc.to(device)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ## Mendefinisikan argumen (dataops) untuk training nanti
    TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
    
    if (re.findall(r'.*/(.*)$', MODEL_NAME) == []): 
        NAME = f'IndoNLI-{DATA_NAME}-with-{str(MODEL_NAME)}'
    else:
        new_name = re.findall(r'.*/(.*)$', MODEL_NAME)[0]
        NAME = f'IndoNLI-{DATA_NAME}-with-{str(new_name)}'

    NAME = f'{NAME}-LR-{LEARNING_RATE}'
    
    SC = f'./results/{NAME}-{TIME_NOW}'
    CHECKPOINT_DIR = f'{SC}/checkpoint/'
    MODEL_DIR = f'{SC}/model/'
    OUTPUT_DIR = f'{SC}/output/'
    METRIC_RESULT_DIR = f'{SC}/metric-result/'
    REPO_NAME = f'fine-tuned-{NAME}'[:96]

    training_args_sc = TrainingArguments(
        
        # Checkpoint
        output_dir=CHECKPOINT_DIR,
        overwrite_output_dir=True,
        save_strategy='steps',
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
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        bf16=False,
        dataloader_num_workers=cpu_count(),
        
        # Miscellaneous
        evaluation_strategy='steps',
        save_steps=int((data_indonli['train'].num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        eval_steps=int((data_indonli['train'].num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        seed=SEED,
        hub_token=HUB_TOKEN,
        push_to_hub=True,
        hub_model_id=REPO_NAME,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    # ## Mulai training untuk fine-tune IndoNLI diatas IndoBERT
    trainer_sc = Trainer(
        model=model_sc,
        args=training_args_sc,
        train_dataset=tokenized_data_indonli_train,
        eval_dataset=tokenized_data_indonli_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer_sc.train()

    # ## Simpan model Sequence Classification
    trainer_sc.save_model(MODEL_DIR)

    # ## Method untuk melihat isi PredictionOutput
    def represent_prediction_output(predict_result):
        predictions_idx = np.argmax(predict_result.predictions, axis=1)
        label_array = np.asarray(predict_result.label_ids)
        
        premise_array = []
        hypothesis_array = []
        
        pred_label_array = []
        gold_label_array = []
        
        for i in tqdm(range(len(predict_result.predictions))):
            
            premise = []
            hypothesis = []
            
            for j in range(len(tokenized_data_indonli_validation[i]['token_type_ids'])):

                if tokenized_data_indonli_validation[i]['token_type_ids'][j] == 0:
                    premise.append(tokenized_data_indonli_validation[i]['input_ids'][j])

                else:
                    hypothesis.append(tokenized_data_indonli_validation[i]['input_ids'][j])
            
            premise_decoded = tokenizer.decode(premise, skip_special_tokens=True)
            hypothesis_decoded = tokenizer.decode(hypothesis, skip_special_tokens=True)

            premise_array.append(premise_decoded)
            hypothesis_array.append(hypothesis_decoded)
            
            pred_label_array.append(predictions_idx[i])
            gold_label_array.append(label_array[i])
            
        id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        nli_df = pd.DataFrame({'Premise': premise_array, 
                                'Hypothesis': hypothesis_array,
                            'Prediction Label': pred_label_array,
                            'Gold Label': gold_label_array
                                })
        
        nli_df["Prediction Label"] = nli_df["Prediction Label"].map(id2label)
        nli_df["Gold Label"] = nli_df["Gold Label"].map(id2label)
        
        return nli_df
    
    # # Melakukan prediksi dari model
    predict_result = trainer_sc.predict(tokenized_data_indonli_validation)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()

    nli_df = represent_prediction_output(predict_result)
    nli_df.to_csv(f'{OUTPUT_DIR}/output_df.csv')
    
    # # Melakukan evaluasi dari prediksi
    metric_result = compute_metrics(predict_result)
    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/metric_result.txt', "w") as f:
        f.write(str(metric_result))
        f.close()

    # # Upload folder ke Hugging Face
    api = HfApi()

    api.upload_folder(
        folder_path=f"{OUTPUT_DIR}",
        repo_id=f"{USER}/{REPO_NAME}",
        repo_type="model",
        token=HUB_TOKEN,
        path_in_repo="results/output",
    )

    api.upload_folder(
        folder_path=f"{METRIC_RESULT_DIR}",
        repo_id=f"{USER}/{REPO_NAME}",
        repo_type="model",
        token=HUB_TOKEN,
        path_in_repo="results/evaluation",
    )


    print(f"Selesai training IndoNLI dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, dan token: {HUB_TOKEN}")
    print("Program training IndoNLI selesai!")