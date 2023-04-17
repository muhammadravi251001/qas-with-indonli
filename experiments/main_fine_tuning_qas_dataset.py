import argparse
import sys
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="Program untuk fine-tuning dataset QA")
parser.add_argument('-m', '--model_name', type=str, metavar='', required=True, help="Nama model Anda; String; choice=[indolem, indonlu, xlmr, your model choice]")
parser.add_argument('-d', '--data_name', type=str, metavar='', required=True, help="Nama dataset Anda; String; choice=[squadid, idkmrc, tydiqaid]")
parser.add_argument('-e', '--epoch', type=int, metavar='', required=True, help="Jumlah epoch Anda; Integer; choice=[all integer]")
parser.add_argument('-sa', '--sample', type=str, metavar='', required=True, help="Jumlah sampling data Anda; Integer; choice=[max, all integer]")
parser.add_argument('-l', '--learn_rate', type=str, metavar='', required=False, help="Jumlah learning rate Anda; Float; choice=[all float]; default=1e-5", default=1e-5)
parser.add_argument('-se', '--seed', type=int, metavar='', required=False, help="Jumlah seed Anda; Integer; choice=[all integer]; default=42", default=42)
parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=False, help="Jumlah batch-size Anda; Integer; choice=[all integer]; default=16", default=16)
parser.add_argument('-ga', '--gradient_accumulation', type=int, metavar='', required=False, help="Jumlah gradient accumulation Anda; Integer; choice=[all integer]; default=8", default=8)
parser.add_argument('-t', '--token', type=str, metavar='', required=False, help="Token Hugging Face Anda; String; choice=[all string token]; default=(TOKEN_HF_muhammadravi251001)", default="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")
parser.add_argument('-wi', '--with_ittl', type=lambda x: bool(strtobool(x)), metavar='', required=False, help="Dengan ITTL atau tidak?; Boolean; choice=[True, False]; default=False", default=False)
parser.add_argument('-fr', '--freeze', type=lambda x: bool(strtobool(x)), metavar='', required=False, help="Dengan ITTL, apa mau freeze layer BERT?; Boolean; choice=[True, False]; default=False", default=False)
args = parser.parse_args()

if __name__ == "__main__":

    base_model = ["indolem", "indonlu", "xlmr"]
    
    if (args.model_name) in base_model:
        if (args.model_name) == "indolem":
            MODEL_NAME = "indolem/indobert-base-uncased"
        elif (args.model_name) == "indonlu":
            MODEL_NAME = "indobenchmark/indobert-large-p2"
        elif (args.model_name) == "xlmr":
            MODEL_NAME = "xlm-roberta-large"
    else: MODEL_NAME = str(args.model_name)
    
    if (args.data_name) == "squadid":
        DATA_NAME = "Squad-ID"
    elif (args.data_name) == "idkmrc":
        DATA_NAME = "IDK-MRC"
    elif (args.data_name) == "tydiqaid":
        DATA_NAME = "TYDI-QA-ID"

    if (args.sample) == "max":
        SAMPLE = sys.maxsize
    else: SAMPLE = int(args.sample)

    EPOCH = int(args.epoch)
    LEARNING_RATE = float(args.learn_rate)
    SEED = int(args.seed)
    HUB_TOKEN = str(args.token)
    BATCH_SIZE = int(args.batch_size)
    GRADIENT_ACCUMULATION = int(args.gradient_accumulation)
    FREEZE = bool(args.freeze)

    if (args.with_ittl) == False:
        MODEL_SC_NAME = None
    else:
        
    # Model-model dibawah dipilih karena memiliki akurasi terbaik dari eksperimen sebelumnya.
        if (args.model_name) == "indolem":
            MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-indobert-base-uncased"
        elif (args.model_name) == "indonlu":
            MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-indobert-large-p2"
        elif (args.model_name) == "xlmr":
            MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-xlm-roberta-base"
        else: MODEL_SC_NAME = str(args.model_name)

    print("Program fine-tuning dataset QA mulai...")
    print(f"Mulai fine-tuning dataset QA dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, freeze: {FREEZE}, model_sc: {MODEL_SC_NAME}, dan token: {HUB_TOKEN}")

    # ## Mendefinisikan hyperparameter
    MODEL_NAME = MODEL_NAME
    EPOCH = EPOCH
    SAMPLE = SAMPLE
    LEARNING_RATE = LEARNING_RATE
    HUB_TOKEN = HUB_TOKEN
    SEED = SEED
    BATCH_SIZE = BATCH_SIZE
    GRADIENT_ACCUMULATION = GRADIENT_ACCUMULATION

    if HUB_TOKEN == "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": USER = "muhammadravi251001"
    else: USER = "afaji"
    
    MODEL_NER_NAME = "cahya/xlm-roberta-base-indonesian-NER"
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
    import transformers
    import evaluate
    import torch
    import operator
    import re
    import sys
    import collections
    import string
    import contextlib

    import numpy as np
    import pandas as pd
    import torch.nn as nn

    from multiprocessing import cpu_count
    from evaluate import load
    from nusacrowd import NusantaraConfigHelper
    from datetime import datetime
    from huggingface_hub import notebook_login
    from tqdm import tqdm
    from IPython.display import display
    from huggingface_hub import HfApi
    
    from datasets import (
        load_dataset, 
        Dataset,
        DatasetDict
    )
    from transformers import (
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        BertForSequenceClassification,
        BertForQuestionAnswering,
        AutoTokenizer,
        EarlyStoppingCallback, 
        AutoModelForQuestionAnswering,
        pipeline
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ## Gunakan tokenizer yang sudah pre-trained
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ## Import dataset QAS
    if (DATA_NAME == "Squad-ID"):
        conhelps = NusantaraConfigHelper()
        data_qas_id = conhelps.filtered(lambda x: 'squad_id' in x.dataset_name)[0].load_dataset()

        df_train = pd.DataFrame(data_qas_id['train'])
        df_test = pd.DataFrame(data_qas_id['validation'])

        cols = ['context', 'question', 'answer']
        new_df_test = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_test['context']))):
            new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                            'question': df_test["question"][i], 
                                            'answer': {"text": eval(df_test["answer"][i][0])['text'], 
                                            "answer_start": eval(df_test["answer"][i][0])['answer_start'], 
                                            "answer_end": eval(df_test["answer"][i][0])['answer_end']}}, 
                                        ignore_index=True)

        cols = ['context', 'question', 'answer']
        new_df_train = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_train['context']))):
            new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                            'question': df_train["question"][i], 
                                            'answer': {"text": eval(df_train["answer"][i][0])['text'], 
                                            "answer_start": eval(df_train["answer"][i][0])['answer_start'], 
                                            "answer_end": eval(df_train["answer"][i][0])['answer_end']}}, 
                                        ignore_index=True)

        train_final_df = new_df_train[:-11874]
        validation_final_df = new_df_train[-11874:]

        train_dataset = Dataset.from_dict(train_final_df)
        validation_dataset = Dataset.from_dict(validation_final_df)
        test_dataset = Dataset.from_dict(df_test)

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    elif (DATA_NAME == "IDK-MRC"):
        conhelps = NusantaraConfigHelper()
        data_qas_id = conhelps.filtered(lambda x: 'idk_mrc' in x.dataset_name)[0].load_dataset()

        df_train = pd.DataFrame(data_qas_id['train'])
        df_validation = pd.DataFrame(data_qas_id['validation'])
        df_test = pd.DataFrame(data_qas_id['test'])

        cols = ['context', 'question', 'answer']
        new_df_train = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_train['context']))):
            for j in df_train["qas"][i]:
                if len(j['answers']) != 0:
                    new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                                        'question': j['question'], 
                                                        'answer': {"text": j['answers'][0]['text'], 
                                                                   "answer_start": j['answers'][0]['answer_start'], 
                                                                   "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                                   ignore_index=True)
                else:
                    new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                                        'question': j['question'], 
                                                        'answer': {"text": str(), 
                                                                   "answer_start": 0, 
                                                                   "answer_end": 0}}, 
                                                                   ignore_index=True)

        cols = ['context', 'question', 'answer']
        new_df_val = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_validation['context']))):
            for j in df_validation["qas"][i]:
                if len(j['answers']) != 0:
                    new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                                    'question': j['question'], 
                                                    'answer': {"text": j['answers'][0]['text'], 
                                                               "answer_start": j['answers'][0]['answer_start'], 
                                                               "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                               ignore_index=True)
                else:
                    new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                                    'question': j['question'], 
                                                    'answer': {"text": str(), 
                                                               "answer_start": 0, 
                                                               "answer_end": 0}}, 
                                                               ignore_index=True)        

        cols = ['context', 'question', 'answer']
        new_df_test = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_test['context']))):
            for j in df_test["qas"][i]:
                if len(j['answers']) != 0:
                    new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                                    'question': j['question'], 
                                                    'answer': {"text": j['answers'][0]['text'], 
                                                               "answer_start": j['answers'][0]['answer_start'], 
                                                               "answer_end": j['answers'][0]['answer_start'] + len(j['answers'][0]['text'])}}, 
                                                               ignore_index=True)
                else:
                    new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                                    'question': j['question'], 
                                                    'answer': {"text": str(), 
                                                               "answer_start": 0, 
                                                               "answer_end": 0}}, 
                                                               ignore_index=True)
        
        train_dataset = Dataset.from_dict(new_df_train)
        validation_dataset = Dataset.from_dict(new_df_val)
        test_dataset = Dataset.from_dict(new_df_test)
        
        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    elif (DATA_NAME == "TYDI-QA-ID"):
        conhelps = NusantaraConfigHelper()
        data_qas_id = conhelps.filtered(lambda x: 'tydiqa_id' in x.dataset_name)[0].load_dataset()

        df_train = pd.DataFrame(data_qas_id['train'])
        df_validation = pd.DataFrame(data_qas_id['validation'])
        df_test = pd.DataFrame(data_qas_id['test'])

        cols = ['context', 'question', 'answer']
        new_df_train = pd.DataFrame(columns=cols)

        for i in range(len(df_train['context'])):
            answer_start = df_train['context'][i].index(df_train['label'][i])
            answer_end = answer_start + len(df_train['label'][i])
            new_df_train = new_df_train.append({'context': df_train["context"][i], 
                                                'question': df_train["question"][i], 
                                                'answer': {"text": df_train["label"][i], 
                                                           "answer_start": answer_start, 
                                                           "answer_end": answer_end}}, 
                                                           ignore_index=True)

        cols = ['context', 'question', 'answer']
        new_df_val = pd.DataFrame(columns=cols)    
            
        for i in range(len(df_validation['context'])):
            answer_start = df_validation['context'][i].index(df_validation['label'][i])
            answer_end = answer_start + len(df_validation['label'][i])
            new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                            'question': df_validation["question"][i], 
                                            'answer': {"text": df_validation["label"][i], 
                                                       "answer_start": answer_start, 
                                                       "answer_end": answer_end}}, 
                                                       ignore_index=True)    
            
        cols = ['context', 'question', 'answer']
        new_df_test = pd.DataFrame(columns=cols)

        for i in range(len(df_test['context'])):
            answer_start = df_test['context'][i].index(df_test['label'][i])
            answer_end = answer_start + len(df_test['label'][i])
            new_df_test = new_df_test.append({'context': df_test["context"][i], 
                                            'question': df_test["question"][i], 
                                            'answer': {"text": df_test["label"][i], 
                                                       "answer_start": answer_start, 
                                                       "answer_end": answer_end}}, 
                                                       ignore_index=True)
        
        train_dataset = Dataset.from_dict(new_df_train)
        validation_dataset = Dataset.from_dict(new_df_val)
        test_dataset = Dataset.from_dict(new_df_test)

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset, "test": test_dataset})

    # ## Fungsi utilitas untuk pre-process dataset QAS
    def rindex(lst, value, operator=operator):
      return len(lst) - operator.indexOf(reversed(lst), value) - 1

    def preprocess_function_qa(examples, tokenizer, MAX_LENGTH=MAX_LENGTH, STRIDE=STRIDE, 
                               rindex=rindex, operator=operator):
        
        examples["question"] = [q.lstrip() for q in examples["question"]]
        examples["context"] = [c.lstrip() for c in examples["context"]]

        if (args.model_name) == "xlmr":
            
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
        
        else:

            tokenized_examples = tokenizer(
                examples['question'],
                examples['context'],
                truncation=True,
                max_length = MAX_LENGTH,
                stride=STRIDE,
                return_overflowing_tokens=True,
                return_token_type_ids=True,
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
            answer = examples['answer'][cur_example_idx]
            answer = eval(str(answer))
            answer_start = answer['answer_start']
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

    # ## Melakukan tokenisasi data IndoNLI
    tokenized_data_qas_id = data_qas_id.map(
        preprocess_function_qa,
        batched=True,
        load_from_cache_file=True,
        num_proc=1,
        remove_columns=data_qas_id['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'MAX_LENGTH': MAX_LENGTH, 
                'STRIDE': STRIDE, 'rindex': rindex, 'operator': operator}
    )

    tokenized_data_qas_id = tokenized_data_qas_id.remove_columns(["offset_mapping", "overflow_to_sample_mapping"])
    
    if (args.model_name) == "xlmr":
        tokenized_data_qas_id.set_format("torch", columns=["input_ids"], output_all_columns=True, device=device)
    
    else:
        tokenized_data_qas_id.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)

    tokenized_data_qas_id_train = Dataset.from_dict(tokenized_data_qas_id["train"][:SAMPLE])
    tokenized_data_qas_id_validation = Dataset.from_dict(tokenized_data_qas_id["validation"][:SAMPLE])
    tokenized_data_qas_id_test = Dataset.from_dict(tokenized_data_qas_id["test"][:SAMPLE])

    # # Tahapan fine-tune dataset QAS diatas model
    # ## Gunakan model Sequence Classification yang sudah pre-trained
    model_qa = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME, num_labels=2)
    model_qa = model_qa.to(device)

    if MODEL_SC_NAME != None:
        # ## Dictionary untuk mapping label
        id2label = {0: 'entailment', 1: 'neutral', 
                    2: 'contradiction'}
        label2id = {'entailment': 0, 'neutral': 
                    1, 'contradiction': 2}
        accuracy = evaluate.load('accuracy')

        model_sc = BertForSequenceClassification.from_pretrained(
            MODEL_SC_NAME, num_labels=3, 
            id2label=id2label, label2id=label2id)
        
        # Bagian transfer-learning-nya
        filtered_dict = {k: v for k, v in model_sc.state_dict().items() if k in model_qa.state_dict()}
        model_qa.state_dict().update(filtered_dict)
        model_qa.load_state_dict(model_qa.state_dict())

    # ## Freeze BERT layer (opsional)
    if FREEZE == True:
        for name, param in model_qa.named_parameters():
            if 'qa_outputs' not in name:
                param._trainable = False
    
    # ## Melakukan pengumpulan data dengan padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # # Melakukan evaluasi dari prediksi
    def normalize_text(s):
        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)
        def white_space_fix(text):
            return " ".join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def compute_f1(pred, gold):
        pred_tokens = normalize_text(pred).split() # True positive + False positive = Untuk precision
        gold_tokens = normalize_text(gold).split() # True positive + False negatives = Untuk recall
        common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
        num_same = sum(common.values()) # True positive
        
        if len(gold_tokens) == 0 or len(pred_tokens) == 0: 
            return int(gold_tokens == pred_tokens)
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(gold_tokens)
        f1 = (2.0 * precision * recall) / (precision + recall)
        
        return f1

    def compute_metrics(predict_result, 
                    tokenized_data_qas_id_validation=tokenized_data_qas_id_validation, 
                    tokenized_data_qas_id_test=tokenized_data_qas_id_test):
    
        predictions_idx = np.argmax(predict_result.predictions, axis=2)
        denominator = len(predictions_idx[0])
        label_array = np.asarray(predict_result.label_ids)
        total_correct = 0
        f1_array = []
        
        if len(predict_result.predictions[0]) == len(tokenized_data_qas_id_validation):
            tokenized_data = tokenized_data_qas_id_validation
        
        elif len(predict_result.predictions[0]) == len(tokenized_data_qas_id_test):
            tokenized_data = tokenized_data_qas_id_test

        for i in range(len(predict_result.predictions[0])):
            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1
            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1

            pred_text = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_pred_idx: end_pred_idx])
            gold_text = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_gold_idx: end_gold_idx])

            if pred_text == gold_text:
                total_correct += 1

            f1 = compute_f1(pred=pred_text, gold=gold_text)

            f1_array.append(f1)

        exact_match = ((total_correct / denominator) * 100.0)
        final_f1 = np.mean(f1_array) * 100.0

        return {'exact_match': exact_match, 'f1': final_f1}

    # ## Mendefinisikan argumen (dataops) untuk training nanti
    TIME_NOW = str(datetime.now()).replace(":", "-").replace(" ", "_").replace(".", "_")
    
    if (re.findall(r'.*/(.*)$', MODEL_NAME) == []): 
        NAME = f'DatasetQAS-{DATA_NAME}-with-{str(MODEL_NAME)}'
    else:
        new_name = re.findall(r'.*/(.*)$', MODEL_NAME)[0]
        NAME = f'DatasetQAS-{DATA_NAME}-with-{str(new_name)}'
    
    if MODEL_SC_NAME == None:
        NAME = f'{NAME}-without-ITTL'
    else:
        NAME = f'{NAME}-with-ITTL'

    if FREEZE == True:
        NAME = f'{NAME}-with-freeze'
    else:
        NAME = f'{NAME}-without-freeze'
    
    NAME = f'{NAME}-LR-{LEARNING_RATE}'
    
    QA = f'./results/{NAME}-{TIME_NOW}'
    CHECKPOINT_DIR = f'{QA}/checkpoint/'
    MODEL_DIR = f'{QA}/model/'
    OUTPUT_DIR = f'{QA}/output/'
    METRIC_RESULT_DIR = f'{QA}/metric-result/'
    REPO_NAME = f'fine-tuned-{NAME}'[:96]

    training_args_qa = TrainingArguments(
        
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
        save_steps=int((tokenized_data_qas_id_train.num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        eval_steps=int((tokenized_data_qas_id_train.num_rows / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * EVAL_STEPS_RATIO),
        seed=SEED,
        hub_token=HUB_TOKEN,
        push_to_hub=True,
        hub_model_id=REPO_NAME,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    # ## Mulai training untuk fine-tune IndoNLI diatas IndoBERT
    trainer_qa = Trainer(
        model=model_qa,
        args=training_args_qa,
        train_dataset=tokenized_data_qas_id_train,
        eval_dataset=tokenized_data_qas_id_validation,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    trainer_qa.train()

    # ## Simpan model Question Answering
    trainer_qa.save_model(MODEL_DIR)

    # ## Assign answer type untuk evaluasi nanti
    nlp_ner = pipeline(task="ner", model=MODEL_NER_NAME, tokenizer=MODEL_NER_NAME)

    def assign_answer_types(answer, nlp=nlp_ner):
    
        if answer == str(): 
            return ["NULL"]
        
        entity_array = []    
        ner_result = nlp(answer)
        
        for i in ner_result:
            entity = i['entity'][2:]
            entity_array.append(entity)
        
        if entity_array == []: 
            return ["NULL"]

        return list(set(entity_array))

    # ## Method untuk melihat isi PredictionOutput
    def represent_prediction_output(predict_result, assign_answer_types=assign_answer_types):
        
        predictions_idx = np.argmax(predict_result.predictions, axis=2)
        label_array = np.asarray(predict_result.label_ids)
            
        question_array = []
        context_array = []

        pred_answer_array = []
        gold_answer_array = []
        
        answer_types_array = []
        
        for i in tqdm(range(len(predict_result.predictions[0]))):

            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1

            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1
            
            if len(predict_result.predictions[0]) == len(tokenized_data_qas_id_validation):
                tokenized_data = tokenized_data_qas_id_validation
            
            elif len(predict_result.predictions[0]) == len(tokenized_data_qas_id_test):
                tokenized_data = tokenized_data_qas_id_test

            pred_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_pred_idx: end_pred_idx], skip_special_tokens=True)

            gold_answer = tokenizer.decode(tokenized_data[i]['input_ids']
                                        [start_gold_idx: end_gold_idx], skip_special_tokens=True)

            pred_answer_array.append(pred_answer)
            gold_answer_array.append(gold_answer)
            
            answer_types_array.append(assign_answer_types(answer=gold_answer))
            
            question = []
            context = []
            
            if (args.model_name) == "xlmr":
                
                start_question = tokenized_data[i]['input_ids'].index(0)
                end_question = tokenized_data[i]['input_ids'].index(2)  + 1
                start_context = end_question

                question.append(tokenized_data[i]['input_ids'][start_question: end_question])
                context.append(tokenized_data[i]['input_ids'][start_context: ])

                question_decoded = tokenizer.decode(question[0], skip_special_tokens=True)
                context_decoded = tokenizer.decode(context[0], skip_special_tokens=True)
                
            else:
                
                for j in range(len(tokenized_data[i]['token_type_ids'])):

                    if tokenized_data[i]['token_type_ids'][j] == 0:
                        question.append(tokenized_data[i]['input_ids'][j])

                    else:
                        context.append(tokenized_data[i]['input_ids'][j])

            question_decoded = tokenizer.decode(question, skip_special_tokens=True)
            context_decoded = tokenizer.decode(context, skip_special_tokens=True)
            
            question_array.append(question_decoded)
            context_array.append(context_decoded)
            
        qas_df = pd.DataFrame({'Context': context_array, 
                                'Question': question_array, 
                                'Prediction Answer': pred_answer_array,
                                'Gold Answer': gold_answer_array,
                                'Answer Type': answer_types_array,
                                'Reasoning Type': '-'
                            })
        
        if DATA_NAME == "Squad-ID": 
            
            # Apa (13)
            qas_df['Reasoning Type'][1283] = 'SSR'
            qas_df['Reasoning Type'][3228] = 'AoI'
            qas_df['Reasoning Type'][4120] = 'SSR'
            qas_df['Reasoning Type'][4456] = 'MSR'
            qas_df['Reasoning Type'][4959] = 'AoI'
            
            qas_df['Reasoning Type'][5458] = 'SSR'
            qas_df['Reasoning Type'][6122] = 'MSR'
            qas_df['Reasoning Type'][6151] = 'AoI'
            qas_df['Reasoning Type'][6795] = 'AoI'
            qas_df['Reasoning Type'][7119] = 'PP' 
            
            qas_df['Reasoning Type'][9685] = 'WM'
            qas_df['Reasoning Type'][10043] = 'AoI'
            qas_df['Reasoning Type'][11173] = 'PP'
            
            # Dimana (12)
            qas_df['Reasoning Type'][969] = 'PP'
            qas_df['Reasoning Type'][1209] = 'PP' 
            qas_df['Reasoning Type'][1296] = 'SSR'
            qas_df['Reasoning Type'][2871] = 'WM'
            qas_df['Reasoning Type'][3163] = 'AoI'
            
            qas_df['Reasoning Type'][3870] = 'SSR'
            qas_df['Reasoning Type'][3922] = 'PP'
            qas_df['Reasoning Type'][5462] = 'AoI'
            qas_df['Reasoning Type'][7263] = 'PP'
            qas_df['Reasoning Type'][7319] = 'PP'
            
            qas_df['Reasoning Type'][9000] = 'PP'
            qas_df['Reasoning Type'][9124] = 'PP'
            
            # Kapan (12)
            qas_df['Reasoning Type'][3195] = 'AoI'
            qas_df['Reasoning Type'][3243] = 'AoI'
            qas_df['Reasoning Type'][4214] = 'PP'
            qas_df['Reasoning Type'][4636] = 'MSR'
            qas_df['Reasoning Type'][7122] = 'AoI' 
            
            qas_df['Reasoning Type'][7445] = 'SSR'
            qas_df['Reasoning Type'][7649] = 'AoI'
            qas_df['Reasoning Type'][9372] = 'SSR'
            qas_df['Reasoning Type'][10211] = 'AoI'
            qas_df['Reasoning Type'][10424] = 'AoI' 
            
            qas_df['Reasoning Type'][10700] = 'PP'
            qas_df['Reasoning Type'][11298] = 'AoI'
            
            # Siapa (13)
            qas_df['Reasoning Type'][1778] = 'AoI'
            qas_df['Reasoning Type'][2810] = 'SSR'
            qas_df['Reasoning Type'][3488] = 'WM' 
            qas_df['Reasoning Type'][4661] = 'MSR'
            qas_df['Reasoning Type'][7307] = 'WM'
            
            qas_df['Reasoning Type'][7481] = 'PP'
            qas_df['Reasoning Type'][7840] = 'AoI'
            qas_df['Reasoning Type'][7849] = 'AoI'
            qas_df['Reasoning Type'][7962] = 'PP'
            qas_df['Reasoning Type'][9634] = 'PP'
            
            qas_df['Reasoning Type'][9976] = 'PP'
            qas_df['Reasoning Type'][11349] = 'SSR'
            qas_df['Reasoning Type'][11367] = 'PP' 
            
            # Kenapa (12)
            qas_df['Reasoning Type'][2723] = 'AoI'
            qas_df['Reasoning Type'][3348] = 'WM'
            qas_df['Reasoning Type'][4390] = 'AoI'
            qas_df['Reasoning Type'][4955] = 'AoI'
            qas_df['Reasoning Type'][5168] = 'SSR' 
            
            qas_df['Reasoning Type'][5728] = 'AoI'
            qas_df['Reasoning Type'][6705] = 'AoI'
            qas_df['Reasoning Type'][7214] = 'AoI'
            qas_df['Reasoning Type'][9379] = 'AoI'
            qas_df['Reasoning Type'][9946] = 'AoI' 
            
            qas_df['Reasoning Type'][10632] = 'PP'
            qas_df['Reasoning Type'][10837] = 'SSR'
            
            # Bagaimana (12)
            qas_df['Reasoning Type'][353] = 'AoI'
            qas_df['Reasoning Type'][1497] = 'MSR'
            qas_df['Reasoning Type'][1883] = 'SSR'
            qas_df['Reasoning Type'][2739] = 'AoI'
            qas_df['Reasoning Type'][3690] = 'MSR'
            
            qas_df['Reasoning Type'][4338] = 'AoI'
            qas_df['Reasoning Type'][4387] = 'AoI'
            qas_df['Reasoning Type'][5392] = 'WM' 
            qas_df['Reasoning Type'][5840] = 'AoI'
            qas_df['Reasoning Type'][7961] = 'AoI'
            
            qas_df['Reasoning Type'][8409] = 'SSR'
            qas_df['Reasoning Type'][10870] = 'AoI'
            
            # Berapa (13)
            qas_df['Reasoning Type'][818] = 'AoI' 
            qas_df['Reasoning Type'][966] = 'AoI'
            qas_df['Reasoning Type'][1035] = 'AoI'
            qas_df['Reasoning Type'][1238] = 'AoI'
            qas_df['Reasoning Type'][1252] = 'PP'
            
            qas_df['Reasoning Type'][1857] = 'PP' 
            qas_df['Reasoning Type'][2853] = 'PP'
            qas_df['Reasoning Type'][3497] = 'SSR'
            qas_df['Reasoning Type'][4144] = 'MSR'
            qas_df['Reasoning Type'][6468] = 'MSR'
            
            qas_df['Reasoning Type'][7267] = 'SSR'
            qas_df['Reasoning Type'][11035] = 'SSR'
            qas_df['Reasoning Type'][11731] = 'SSR'
            
            # Lainnya (13)
            qas_df['Reasoning Type'][1488] = 'AoI'
            qas_df['Reasoning Type'][1571] = 'AoI'
            qas_df['Reasoning Type'][3093] = 'PP' 
            qas_df['Reasoning Type'][5552] = 'WM'
            qas_df['Reasoning Type'][6256] = 'AoI'
            
            qas_df['Reasoning Type'][6371] = 'MSR'
            qas_df['Reasoning Type'][6672] = 'SSR'
            qas_df['Reasoning Type'][7258] = 'SSR' 
            qas_df['Reasoning Type'][7562] = 'SSR'
            qas_df['Reasoning Type'][8154] = 'MSR'
            
            qas_df['Reasoning Type'][8337] = 'SSR'
            qas_df['Reasoning Type'][9160] = 'AoI'
            qas_df['Reasoning Type'][11621] = 'AoI' 
        
        elif DATA_NAME == "IDK-MRC":
            
            # Apa (14)
            qas_df['Reasoning Type'][66] = 'AoI'
            qas_df['Reasoning Type'][84] = 'AoI'
            qas_df['Reasoning Type'][190] = 'MSR'
            qas_df['Reasoning Type'][207] = 'WM'
            qas_df['Reasoning Type'][214] = 'AoI'
            
            qas_df['Reasoning Type'][320] = 'SSR'
            qas_df['Reasoning Type'][322] = 'AoI'
            qas_df['Reasoning Type'][347] = 'AoI'
            qas_df['Reasoning Type'][363] = 'PP'
            qas_df['Reasoning Type'][372] = 'AoI'
            
            qas_df['Reasoning Type'][490] = 'PP'
            qas_df['Reasoning Type'][566] = 'PP'
            qas_df['Reasoning Type'][666] = 'AoI'
            qas_df['Reasoning Type'][732] = 'AoI'
            
            # Dimana (13)
            qas_df['Reasoning Type'][61] = 'AoI'
            qas_df['Reasoning Type'][220] = 'AoI'
            qas_df['Reasoning Type'][222] = 'AoI'
            qas_df['Reasoning Type'][227] = 'AoI'
            qas_df['Reasoning Type'][294] = 'AoI'
            
            qas_df['Reasoning Type'][378] = 'AoI'
            qas_df['Reasoning Type'][393] = 'MSR'
            qas_df['Reasoning Type'][394] = 'AoI'
            qas_df['Reasoning Type'][506] = 'AoI'
            qas_df['Reasoning Type'][525] = 'WM'
            
            qas_df['Reasoning Type'][729] = 'WM'
            qas_df['Reasoning Type'][730] = 'AoI'
            qas_df['Reasoning Type'][763] = 'AoI'
            
            # Kapan (14)
            qas_df['Reasoning Type'][88] = 'AoI'
            qas_df['Reasoning Type'][210] = 'AoI'
            qas_df['Reasoning Type'][221] = 'AoI'
            qas_df['Reasoning Type'][228] = 'WM'
            qas_df['Reasoning Type'][312] = 'SSR'
            
            qas_df['Reasoning Type'][385] = 'PP'
            qas_df['Reasoning Type'][391] = 'MSR'
            qas_df['Reasoning Type'][421] = 'AoI'
            qas_df['Reasoning Type'][514] = 'AoI'
            qas_df['Reasoning Type'][533] = 'AoI'
            
            qas_df['Reasoning Type'][540] = 'MSR'
            qas_df['Reasoning Type'][580] = 'AoI'
            qas_df['Reasoning Type'][657] = 'WM'
            qas_df['Reasoning Type'][809] = 'MSR'
            
            # Siapa (13)
            qas_df['Reasoning Type'][23] = 'AoI'
            qas_df['Reasoning Type'][79] = 'AoI'
            qas_df['Reasoning Type'][120] = 'AoI'
            qas_df['Reasoning Type'][269] = 'AoI'
            qas_df['Reasoning Type'][425] = 'AoI'
            
            qas_df['Reasoning Type'][449] = 'AoI'
            qas_df['Reasoning Type'][543] = 'AoI'
            qas_df['Reasoning Type'][551] = 'AoI'
            qas_df['Reasoning Type'][618] = 'AoI'
            qas_df['Reasoning Type'][646] = 'MSR'
            
            qas_df['Reasoning Type'][741] = 'MSR'
            qas_df['Reasoning Type'][751] = 'WM'
            qas_df['Reasoning Type'][775] = 'PP'
            
            # Kenapa (8)
            qas_df['Reasoning Type'][18] = 'MSR'
            qas_df['Reasoning Type'][19] = 'AoI'
            qas_df['Reasoning Type'][54] = 'MSR'
            qas_df['Reasoning Type'][55] = 'AoI'
            qas_df['Reasoning Type'][145] = 'AoI'
            
            qas_df['Reasoning Type'][413] = 'AoI'
            qas_df['Reasoning Type'][675] = 'AoI'
            qas_df['Reasoning Type'][832] = 'AoI'
            
            # Bagaimana (12)
            qas_df['Reasoning Type'][44] = 'AoI'
            qas_df['Reasoning Type'][286] = 'AoI'
            qas_df['Reasoning Type'][455] = 'AoI'
            qas_df['Reasoning Type'][535] = 'AoI'
            qas_df['Reasoning Type'][612] = 'MSR'
            
            qas_df['Reasoning Type'][613] = 'AoI'
            qas_df['Reasoning Type'][649] = 'AoI'
            qas_df['Reasoning Type'][753] = 'AoI'
            qas_df['Reasoning Type'][757] = 'AoI'
            qas_df['Reasoning Type'][794] = 'SSR'
            
            qas_df['Reasoning Type'][795] = 'AoI'
            qas_df['Reasoning Type'][839] = 'AoI'
            
            # Berapa (13)
            qas_df['Reasoning Type'][20] = 'SSR'
            qas_df['Reasoning Type'][62] = 'MSR'
            qas_df['Reasoning Type'][104] = 'AoI'
            qas_df['Reasoning Type'][107] = 'AoI'
            qas_df['Reasoning Type'][265] = 'AoI'
            
            qas_df['Reasoning Type'][434] = 'MSR'
            qas_df['Reasoning Type'][581] = 'AoI'
            qas_df['Reasoning Type'][614] = 'MSR'
            qas_df['Reasoning Type'][696] = 'AoI'
            qas_df['Reasoning Type'][697] = 'AoI'

            qas_df['Reasoning Type'][698] = 'MSR'
            qas_df['Reasoning Type'][724] = 'AoI'
            qas_df['Reasoning Type'][781] = 'PP'

            # Lainnya (14)
            qas_df['Reasoning Type'][35] = 'WM'
            qas_df['Reasoning Type'][38] = 'AoI'
            qas_df['Reasoning Type'][98] = 'SSR'
            qas_df['Reasoning Type'][177] = 'AoI'
            qas_df['Reasoning Type'][198] = 'AoI'

            qas_df['Reasoning Type'][375] = 'SSR'
            qas_df['Reasoning Type'][384] = 'AoI'
            qas_df['Reasoning Type'][412] = 'PP'
            qas_df['Reasoning Type'][442] = 'AoI'
            qas_df['Reasoning Type'][444] = 'MSR'

            qas_df['Reasoning Type'][450] = 'PP'
            qas_df['Reasoning Type'][602] = 'PP'
            qas_df['Reasoning Type'][640] = 'MSR'
            qas_df['Reasoning Type'][768] = 'AoI'
        
        elif DATA_NAME == "TYDI-QA-ID":
            
            # Apa (15)
            qas_df['Reasoning Type'][23] = 'MSR'
            qas_df['Reasoning Type'][32] = 'SSR'
            qas_df['Reasoning Type'][129] = 'PP'
            qas_df['Reasoning Type'][158] = 'MSR'
            qas_df['Reasoning Type'][193] = 'MSR'
            
            qas_df['Reasoning Type'][332] = 'PP'
            qas_df['Reasoning Type'][334] = 'PP'
            qas_df['Reasoning Type'][427] = 'WM'
            qas_df['Reasoning Type'][451] = 'PP'
            qas_df['Reasoning Type'][469] = 'PP' 
            
            qas_df['Reasoning Type'][474] = 'PP'
            qas_df['Reasoning Type'][537] = 'SSR'
            qas_df['Reasoning Type'][619] = 'MSR'
            qas_df['Reasoning Type'][624] = 'PP'
            qas_df['Reasoning Type'][808] = 'PP' 
            
            # Dimana (14)
            qas_df['Reasoning Type'][3] = 'MSR'
            qas_df['Reasoning Type'][66] = 'PP'
            qas_df['Reasoning Type'][163] = 'PP'
            qas_df['Reasoning Type'][164] = 'SSR'
            qas_df['Reasoning Type'][296] = 'AoI'
            
            qas_df['Reasoning Type'][371] = 'MSR'
            qas_df['Reasoning Type'][431] = 'AoI'
            qas_df['Reasoning Type'][437] = 'WM'
            qas_df['Reasoning Type'][489] = 'AoI'
            qas_df['Reasoning Type'][519] = 'SSR'
            
            qas_df['Reasoning Type'][607] = 'SSR'
            qas_df['Reasoning Type'][625] = 'PP'
            qas_df['Reasoning Type'][668] = 'WM'
            qas_df['Reasoning Type'][757] = 'WM'
            
            # Kapan (15)
            qas_df['Reasoning Type'][57] = 'MSR' 
            qas_df['Reasoning Type'][89] = 'MSR'
            qas_df['Reasoning Type'][123] = 'AoI'
            qas_df['Reasoning Type'][179] = 'AoI'
            qas_df['Reasoning Type'][228] = 'SSR'
            
            qas_df['Reasoning Type'][253] = 'SSR' 
            qas_df['Reasoning Type'][279] = 'PP'
            qas_df['Reasoning Type'][280] = 'AoI'
            qas_df['Reasoning Type'][340] = 'MSR'
            qas_df['Reasoning Type'][386] = 'SSR'
            
            qas_df['Reasoning Type'][404] = 'SSR' 
            qas_df['Reasoning Type'][429] = 'PP'
            qas_df['Reasoning Type'][484] = 'PP'
            qas_df['Reasoning Type'][529] = 'MSR'
            qas_df['Reasoning Type'][824] = 'MSR'
            
            # Siapa (15)
            qas_df['Reasoning Type'][1] = 'AoI'
            qas_df['Reasoning Type'][12] = 'PP'
            qas_df['Reasoning Type'][30] = 'AoI'
            qas_df['Reasoning Type'][63] = 'SSR'
            qas_df['Reasoning Type'][138] = 'MSR'
            
            qas_df['Reasoning Type'][247] = 'MSR' 
            qas_df['Reasoning Type'][293] = 'AoI'
            qas_df['Reasoning Type'][361] = 'AoI'
            qas_df['Reasoning Type'][393] = 'PP'
            qas_df['Reasoning Type'][546] = 'SSR'
            
            qas_df['Reasoning Type'][548] = 'PP' 
            qas_df['Reasoning Type'][572] = 'PP'
            qas_df['Reasoning Type'][715] = 'PP'
            qas_df['Reasoning Type'][805] = 'PP'
            qas_df['Reasoning Type'][843] = 'PP'
            
            # Kenapa (6)
            qas_df['Reasoning Type'][109] = 'MSR' 
            qas_df['Reasoning Type'][248] = 'WM'
            qas_df['Reasoning Type'][432] = 'MSR'
            qas_df['Reasoning Type'][565] = 'SSR'
            qas_df['Reasoning Type'][597] = 'AoI'
            
            qas_df['Reasoning Type'][771] = 'MSR'
            
            # Bagaimana (5)
            qas_df['Reasoning Type'][93] = 'AoI'
            qas_df['Reasoning Type'][133] = 'SSR'
            qas_df['Reasoning Type'][151] = 'PP'
            qas_df['Reasoning Type'][312] = 'AoI'
            qas_df['Reasoning Type'][390] = 'PP' 
            
            # Berapa (15)
            qas_df['Reasoning Type'][54] = 'MSR'
            qas_df['Reasoning Type'][127] = 'SSR'
            qas_df['Reasoning Type'][178] = 'MSR'
            qas_df['Reasoning Type'][185] = 'AoI'
            qas_df['Reasoning Type'][205] = 'WM' 
            
            qas_df['Reasoning Type'][241] = 'PP'
            qas_df['Reasoning Type'][346] = 'PP'
            qas_df['Reasoning Type'][350] = 'WM'
            qas_df['Reasoning Type'][418] = 'PP'
            qas_df['Reasoning Type'][430] = 'WM' 
            
            qas_df['Reasoning Type'][512] = 'MSR'
            qas_df['Reasoning Type'][596] = 'PP'
            qas_df['Reasoning Type'][634] = 'PP'
            qas_df['Reasoning Type'][690] = 'SSR'
            qas_df['Reasoning Type'][756] = 'SSR'
            
            # Lainnya (15)
            qas_df['Reasoning Type'][80] = 'PP'
            qas_df['Reasoning Type'][116] = 'MSR'
            qas_df['Reasoning Type'][165] = 'AoI'
            qas_df['Reasoning Type'][319] = 'AoI'
            qas_df['Reasoning Type'][388] = 'MSR' 
            
            qas_df['Reasoning Type'][498] = 'MSR'
            qas_df['Reasoning Type'][507] = 'SSR'
            qas_df['Reasoning Type'][582] = 'PP'
            qas_df['Reasoning Type'][593] = 'AoI'
            qas_df['Reasoning Type'][595] = 'MSR' 
            
            qas_df['Reasoning Type'][702] = 'PP'
            qas_df['Reasoning Type'][709] = 'PP'
            qas_df['Reasoning Type'][750] = 'MSR'
            qas_df['Reasoning Type'][776] = 'SSR'
            qas_df['Reasoning Type'][816] = 'WM'          
        
        assert len(predict_result.predictions[0]) == len(qas_df), "Jumlah prediksi berbeda dengan jumlah evaluasi"
        
        return qas_df

    # # Melakukan prediksi dari model
    predict_result = trainer_qa.predict(tokenized_data_qas_id_test)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()

    qas_df = represent_prediction_output(predict_result)
    qas_df.to_csv(f'{OUTPUT_DIR}/output_df.csv')

    ## Evaluasi umum yang berhubungan dengan EDA
    def general_evaluation(df):
    
        num_apa_right = 0
        num_dimana_right = 0
        num_kapan_right = 0
        num_siapa_right = 0
        num_bagaimana_right = 0
        num_kenapa_right = 0
        num_berapa_right = 0
        num_others_right = 0

        num_apa_wrong = 0
        num_dimana_wrong = 0
        num_kapan_wrong = 0
        num_siapa_wrong = 0
        num_bagaimana_wrong = 0
        num_kenapa_wrong = 0
        num_berapa_wrong = 0
        num_others_wrong = 0

        under_hundred_right = 0
        _101_to_150_right = 0
        _151_to_200_right = 0
        _201_to_250_right = 0
        _251_to_300_right = 0
        _over_301_right = 0

        under_hundred_wrong = 0
        _101_to_150_wrong = 0
        _151_to_200_wrong = 0
        _201_to_250_wrong = 0
        _251_to_300_wrong = 0
        _over_301_wrong = 0

        q_one_to_five_right = 0
        q_six_to_ten_right = 0
        q_eleven_to_fifteen_right = 0
        q_sixteen_to_twenty_right = 0
        q_over_twenty_right = 0

        q_one_to_five_wrong = 0
        q_six_to_ten_wrong = 0
        q_eleven_to_fifteen_wrong = 0
        q_sixteen_to_twenty_wrong = 0
        q_over_twenty_wrong = 0

        a_zero_right = 0
        a_one_to_five_right = 0
        a_six_to_ten_right = 0
        a_eleven_to_fifteen_right = 0
        a_sixteen_to_twenty_right = 0
        a_over_twenty_right = 0

        a_zero_wrong = 0
        a_one_to_five_wrong = 0
        a_six_to_ten_wrong = 0
        a_eleven_to_fifteen_wrong = 0
        a_sixteen_to_twenty_wrong = 0
        a_over_twenty_wrong = 0
        
        num_Person_right = 0
        num_NORP_right = 0
        num_Facility_right = 0
        num_Organization_right = 0
        num_Geo_Political_Entity_right = 0
        num_Location_right = 0
        num_Product_right = 0
        num_Event_right = 0
        num_Work_of_Art_right = 0
        num_Law_right = 0
        num_Language_right = 0
        num_Date_right = 0
        num_Time_right = 0
        num_Percent_right = 0
        num_Money_right = 0
        num_Quantity_right = 0
        num_Ordinal_right = 0
        num_Cardinal_right = 0
        num_null_right = 0
        num_REG_right = 0
        
        num_Person_wrong = 0
        num_NORP_wrong = 0
        num_Facility_wrong = 0
        num_Organization_wrong = 0
        num_Geo_Political_Entity_wrong = 0
        num_Location_wrong = 0
        num_Product_wrong = 0
        num_Event_wrong = 0
        num_Work_of_Art_wrong = 0
        num_Law_wrong = 0
        num_Language_wrong = 0
        num_Date_wrong = 0
        num_Time_wrong = 0
        num_Percent_wrong = 0
        num_Money_wrong = 0
        num_Quantity_wrong = 0
        num_Ordinal_wrong = 0
        num_Cardinal_wrong = 0
        num_null_wrong = 0
        num_REG_wrong = 0
        
        denominator_answer_type = 0
        
        num_wm_right = 0
        num_pp_right = 0
        num_ssr_right = 0
        num_msr_right = 0
        num_aoi_right = 0

        num_wm_wrong = 0
        num_pp_wrong = 0
        num_ssr_wrong = 0
        num_msr_wrong = 0
        num_aoi_wrong = 0

        NUM_REASONING_TYPE_ANNOTATED = 100

        # Cek semua properti EDA, yang berhasil berapa, yang gagal berapa?
        for i in range(len(df)):

            pred_answer = df["Prediction Answer"][i]      
            gold_text = df["Gold Answer"][i]
            current_question = df["Question"][i].split()
            len_current_passage = len(df["Context"][i].split())
            len_current_question = len(df["Question"][i].split())
            len_current_gold_text = len(df["Gold Answer"][i].split())
            reasoning_type = df['Reasoning Type'][i]

            for answer_type in df['Answer Type'][i]:
                denominator_answer_type += 1
                
                if (pred_answer == gold_text):
                    if answer_type == 'PER': num_Person_right += 1
                    elif answer_type == 'NOR': num_NORP_right += 1
                    elif answer_type == 'FAC': num_Facility_right += 1
                    elif answer_type == 'ORG': num_Organization_right += 1
                    elif answer_type == 'GPE': num_Geo_Political_Entity_right += 1
                    elif answer_type == 'LOC': num_Location_right += 1
                    elif answer_type == 'PRD': num_Product_right += 1
                    elif answer_type == 'EVT': num_Event_right += 1
                    elif answer_type == 'WOA': num_Work_of_Art_right += 1
                    elif answer_type == 'LAW': num_Law_right += 1
                    elif answer_type == 'LAN': num_Language_right += 1
                    elif answer_type == 'DAT': num_Date_right += 1
                    elif answer_type == 'TIM': num_Time_right += 1
                    elif answer_type == 'PRC': num_Percent_right += 1
                    elif answer_type == 'MON': num_Money_right += 1
                    elif answer_type == 'QTY': num_Quantity_right += 1
                    elif answer_type == 'ORD': num_Ordinal_right += 1
                    elif answer_type == 'CRD': num_Cardinal_right += 1
                    elif answer_type == 'REG': num_REG_right += 1
                    elif answer_type == 'NULL': num_null_right += 1
                
                elif (pred_answer != gold_text):
                    if answer_type == 'PER': num_Person_wrong += 1
                    elif answer_type == 'NOR': num_NORP_wrong += 1
                    elif answer_type == 'FAC': num_Facility_wrong += 1
                    elif answer_type == 'ORG': num_Organization_wrong += 1
                    elif answer_type == 'GPE': num_Geo_Political_Entity_wrong += 1
                    elif answer_type == 'LOC': num_Location_wrong += 1
                    elif answer_type == 'PRD': num_Product_wrong += 1
                    elif answer_type == 'EVT': num_Event_wrong += 1
                    elif answer_type == 'WOA': num_Work_of_Art_wrong += 1
                    elif answer_type == 'LAW': num_Law_wrong += 1
                    elif answer_type == 'LAN': num_Language_wrong += 1
                    elif answer_type == 'DAT': num_Date_wrong += 1
                    elif answer_type == 'TIM': num_Time_wrong += 1
                    elif answer_type == 'PRC': num_Percent_wrong += 1
                    elif answer_type == 'MON': num_Money_wrong += 1
                    elif answer_type == 'QTY': num_Quantity_wrong += 1
                    elif answer_type == 'ORD': num_Ordinal_wrong += 1
                    elif answer_type == 'CRD': num_Cardinal_wrong += 1
                    elif answer_type == 'REG': num_REG_wrong += 1
                    elif answer_type == 'NULL': num_null_wrong += 1
            
            if (pred_answer == gold_text):
                if 'Apa' in current_question: num_apa_right += 1
                elif 'Apakah' in current_question: num_apa_right += 1
                elif 'apa' in current_question: num_apa_right += 1
                elif 'apakah' in current_question: num_apa_right += 1

                elif 'Dimana' in current_question: num_dimana_right += 1
                elif 'dimana' in current_question: num_dimana_right += 1
                elif 'mana' in current_question: num_dimana_right += 1

                elif 'Kapan' in current_question: num_kapan_right += 1
                elif 'kapan' in current_question: num_kapan_right += 1

                elif 'Siapa' in current_question: num_siapa_right += 1
                elif 'siapa' in current_question: num_siapa_right += 1

                elif 'Bagaimana' in current_question: num_bagaimana_right += 1
                elif 'bagaimana' in current_question: num_bagaimana_right += 1

                elif 'Mengapa' in current_question: num_kenapa_right += 1
                elif 'Kenapa' in current_question: num_kenapa_right += 1
                elif 'mengapa' in current_question: num_kenapa_right += 1
                elif 'kenapa' in current_question: num_kenapa_right += 1

                elif 'Berapa' in current_question: num_berapa_right += 1
                elif 'Berapakah' in current_question: num_berapa_right += 1
                elif 'berapa' in current_question: num_berapa_right += 1
                elif 'berapakah' in current_question: num_berapa_right += 1

                else: num_others_right += 1

                if len_current_passage <= 100: 
                    under_hundred_right += 1
                elif len_current_passage >= 101 & len_current_passage <= 150:
                    _101_to_150_right += 1
                elif len_current_passage >= 151 & len_current_passage <= 200:
                    _151_to_200_right += 1
                elif len_current_passage >= 201 & len_current_passage <= 250:
                    _201_to_250_right += 1
                elif len_current_passage >= 251 & len_current_passage <= 300:
                    _251_to_300_right += 1
                elif len_current_passage >= 301:
                    _over_301_right += 1

                if len_current_question <= 5: 
                    q_one_to_five_right += 1
                elif len_current_question >= 6 & len_current_question <= 10:
                    q_six_to_ten_right += 1
                elif len_current_question >= 11 & len_current_question <= 15:
                    q_eleven_to_fifteen_right += 1
                elif len_current_question >= 16 & len_current_question <= 20:
                    q_sixteen_to_twenty_right += 1
                elif len_current_question >= 21: 
                    q_over_twenty_right += 1

                if len_current_gold_text <= 5: 
                    a_one_to_five_right += 1
                elif len_current_gold_text >= 6 & len_current_gold_text <= 10:
                    a_six_to_ten_right += 1
                elif len_current_gold_text >= 11 & len_current_gold_text <= 15:
                    a_eleven_to_fifteen_right += 1
                elif len_current_gold_text >= 16 & len_current_gold_text <= 20:
                    a_sixteen_to_twenty_right += 1
                elif len_current_gold_text >= 21: 
                    a_over_twenty_right += 1
                elif len_current_gold_text == 0:
                    a_zero_right += 1
                    
                if reasoning_type == "WM": num_wm_right += 1
                elif reasoning_type == "PP": num_pp_right += 1
                elif reasoning_type == "SSR": num_ssr_right += 1
                elif reasoning_type == "MSR": num_msr_right += 1
                elif reasoning_type == "AoI": num_aoi_right += 1

            elif (pred_answer != gold_text):
                if 'Apa' in current_question: num_apa_wrong += 1
                elif 'Apakah' in current_question: num_apa_wrong += 1
                elif 'apa' in current_question: num_apa_wrong += 1
                elif 'apakah' in current_question: num_apa_wrong += 1

                elif 'Dimana' in current_question: num_dimana_wrong += 1
                elif 'dimana' in current_question: num_dimana_wrong += 1
                elif 'mana' in current_question: num_dimana_wrong += 1

                elif 'Kapan' in current_question: num_kapan_wrong += 1
                elif 'kapan' in current_question: num_kapan_wrong += 1

                elif 'Siapa' in current_question: num_siapa_wrong += 1
                elif 'siapa' in current_question: num_siapa_wrong += 1

                elif 'Bagaimana' in current_question: num_bagaimana_wrong += 1
                elif 'bagaimana' in current_question: num_bagaimana_wrong += 1

                elif 'Mengapa' in current_question: num_kenapa_wrong += 1
                elif 'Kenapa' in current_question: num_kenapa_wrong += 1
                elif 'mengapa' in current_question: num_kenapa_wrong += 1
                elif 'kenapa' in current_question: num_kenapa_wrong += 1

                elif 'Berapa' in current_question: num_berapa_wrong += 1
                elif 'Berapakah' in current_question: num_berapa_wrong += 1
                elif 'berapa' in current_question: num_berapa_wrong += 1
                elif 'berapakah' in current_question: num_berapa_wrong += 1

                else: num_others_wrong += 1

                if len_current_passage <= 100: 
                    under_hundred_wrong += 1
                elif len_current_passage >= 101 & len_current_passage <= 150:
                    _101_to_150_wrong += 1
                elif len_current_passage >= 151 & len_current_passage <= 200:
                    _151_to_200_wrong += 1
                elif len_current_passage >= 201 & len_current_passage <= 250:
                    _201_to_250_wrong += 1
                elif len_current_passage >= 251 & len_current_passage <= 300:
                    _251_to_300_wrong += 1
                elif len_current_passage >= 301:
                    _over_301_wrong += 1

                if len_current_question <= 5: 
                    q_one_to_five_wrong += 1
                elif len_current_question >= 6 & len_current_question <= 10:
                    q_six_to_ten_wrong += 1
                elif len_current_question >= 11 & len_current_question <= 15:
                    q_eleven_to_fifteen_wrong += 1
                elif len_current_question >= 16 & len_current_question <= 20:
                    q_sixteen_to_twenty_wrong += 1
                elif len_current_question >= 21: 
                    q_over_twenty_wrong += 1

                if len_current_gold_text <= 5: 
                    a_one_to_five_wrong += 1
                elif len_current_gold_text >= 6 & len_current_gold_text <= 10:
                    a_six_to_ten_wrong += 1
                elif len_current_gold_text >= 11 & len_current_gold_text <= 15:
                    a_eleven_to_fifteen_wrong += 1
                elif len_current_gold_text >= 16 & len_current_gold_text <= 20:
                    a_sixteen_to_twenty_wrong += 1
                elif len_current_gold_text >= 21: 
                    a_over_twenty_wrong += 1
                elif len_current_gold_text == 0:
                    a_zero_wrong += 1
                    
                if reasoning_type == "WM": num_wm_wrong += 1
                elif reasoning_type == "PP": num_pp_wrong += 1
                elif reasoning_type == "SSR": num_ssr_wrong += 1
                elif reasoning_type == "MSR": num_msr_wrong += 1
                elif reasoning_type == "AoI": num_aoi_wrong += 1

        assert len(df) == num_apa_right+num_dimana_right+num_kapan_right+num_siapa_right+\
                            num_bagaimana_right+num_kenapa_right+num_berapa_right+num_others_right+\
                            num_apa_wrong+num_dimana_wrong+num_kapan_wrong+num_siapa_wrong+\
                            num_bagaimana_wrong+num_kenapa_wrong+num_berapa_wrong+num_others_wrong

        assert len(df) == under_hundred_right+_101_to_150_right+_151_to_200_right+_201_to_250_right+\
                            _251_to_300_right+_over_301_right+\
                            under_hundred_wrong+_101_to_150_wrong+_151_to_200_wrong+_201_to_250_wrong+\
                            _251_to_300_wrong+_over_301_wrong

        assert len(df) == q_one_to_five_right+q_six_to_ten_right+q_eleven_to_fifteen_right+q_sixteen_to_twenty_right+\
                            q_over_twenty_right+\
                            q_one_to_five_wrong+q_six_to_ten_wrong+q_eleven_to_fifteen_wrong+q_sixteen_to_twenty_wrong+\
                            q_over_twenty_wrong

        assert len(df) == a_one_to_five_right+a_six_to_ten_right+a_eleven_to_fifteen_right+a_sixteen_to_twenty_right+\
                            a_over_twenty_right+a_zero_right+\
                            a_one_to_five_wrong+a_six_to_ten_wrong+a_eleven_to_fifteen_wrong+a_sixteen_to_twenty_wrong+\
                            a_over_twenty_wrong+a_zero_wrong
        
        assert denominator_answer_type == num_Person_right + num_NORP_right + num_Facility_right + num_Organization_right + num_Geo_Political_Entity_right + \
                                num_Location_right + num_Product_right + num_Event_right + num_Work_of_Art_right + num_Law_right + \
                                num_Language_right + num_Date_right + num_Time_right + num_Percent_right + num_Money_right + \
                                num_Quantity_right + num_Ordinal_right + num_Cardinal_right + num_null_right + num_REG_right + \
                                num_Person_wrong + num_NORP_wrong + num_Facility_wrong + num_Organization_wrong + num_Geo_Political_Entity_wrong + \
                                num_Location_wrong + num_Product_wrong + num_Event_wrong + num_Work_of_Art_wrong + num_Law_wrong + \
                                num_Language_wrong + num_Date_wrong + num_Time_wrong + num_Percent_wrong + num_Money_wrong + \
                                num_Quantity_wrong + num_Ordinal_wrong + num_Cardinal_wrong + num_null_wrong + num_REG_wrong
        
        assert NUM_REASONING_TYPE_ANNOTATED == num_wm_right + num_pp_right + num_ssr_right + num_msr_right + num_aoi_right + \
                    num_wm_wrong + num_pp_wrong + num_ssr_wrong + num_msr_wrong + num_aoi_wrong
        
        print("--- Bagian tentang question type ---")
        print(f"-- Bagian tentang question type yang terprediksi BENAR --")
        print(f"Banyak pertanyaan APA: {num_apa_right}, sebesar: {round((num_apa_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan DIMANA: {num_dimana_right}, sebesar: {round((num_dimana_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan KAPAN: {num_kapan_right}, sebesar: {round((num_kapan_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan SIAPA: {num_siapa_right}, sebesar: {round((num_siapa_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan BAGAIMANA: {num_bagaimana_right}, sebesar: {round((num_bagaimana_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan KENAPA: {num_kenapa_right}, sebesar: {round((num_kenapa_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan BERAPA: {num_berapa_right}, sebesar: {round((num_berapa_right/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan LAINNYA: {num_others_right}, sebesar: {round((num_others_right/len(df) * 100), 2)} %")
        print()
        print(f"-- Bagian tentang question type yang terprediksi SALAH --")
        print(f"Banyak pertanyaan APA: {num_apa_wrong}, sebesar: {round((num_apa_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan DIMANA: {num_dimana_wrong}, sebesar: {round((num_dimana_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan KAPAN: {num_kapan_wrong}, sebesar: {round((num_kapan_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan SIAPA: {num_siapa_wrong}, sebesar: {round((num_siapa_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan BAGAIMANA: {num_bagaimana_wrong}, sebesar: {round((num_bagaimana_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan KENAPA: {num_kenapa_wrong}, sebesar: {round((num_kenapa_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan BERAPA: {num_berapa_wrong}, sebesar: {round((num_berapa_wrong/len(df) * 100), 2)} %")
        print(f"Banyak pertanyaan LAINNYA: {num_others_wrong}, sebesar: {round((num_others_wrong/len(df) * 100), 2)} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Banyak pertanyaan APA yang terpediksi benar sebesar: {round((num_apa_right/(num_apa_right+num_apa_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan DIMANA yang terpediksi benar sebesar: {round((num_dimana_right/(num_dimana_right+num_dimana_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan KAPAN yang terpediksi benar sebesar: {round((num_kapan_right/(num_kapan_right+num_kapan_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan SIAPA yang terpediksi benar sebesar: {round((num_siapa_right/(num_siapa_right+num_siapa_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan BAGAIMANA yang terpediksi benar sebesar: {round((num_bagaimana_right/(num_bagaimana_right+num_bagaimana_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan KENAPA yang terpediksi benar sebesar: {round((num_kenapa_right/(num_kenapa_right+num_kenapa_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan BERAPA yang terpediksi benar sebesar: {round((num_berapa_right/(num_berapa_right+num_berapa_wrong) * 100), 2)} %")
        print(f"Banyak pertanyaan LAINNYA yang terpediksi benar sebesar: {round((num_others_right/(num_others_right+num_others_wrong) * 100), 2)} %")
        print()

        print("--- Bagian tentang panjang context ---")
        print(f"-- Bagian tentang panjang context yang terprediksi BENAR --")
        print(f"Panjang konteks < 100: {under_hundred_right}, sebesar: {round((under_hundred_right/len(df) * 100), 2)} %")
        print(f"Panjang konteks 101 <= x <= 150: {_101_to_150_right}, sebesar: {round((_101_to_150_right/len(df) * 100), 2)} %")
        print(f"Panjang konteks 151 <= x <= 200: {_151_to_200_right}, sebesar: {round((_151_to_200_right/len(df) * 100), 2)} %")
        print(f"Panjang konteks 201 <= x <= 250: {_201_to_250_right}, sebesar: {round((_201_to_250_right/len(df) * 100), 2)} %")
        print(f"Panjang konteks 251 <= x <= 300: {_251_to_300_right}, sebesar: {round((_251_to_300_right/len(df) * 100), 2)} %")
        print(f"Panjang konteks > 300: {_over_301_right}, sebesar: {round((_over_301_right/len(df) * 100), 2)} %")
        print()
        print(f"-- Bagian tentang panjang context yang terprediksi SALAH --")
        print(f"Panjang konteks < 100: {under_hundred_wrong}, sebesar: {round((under_hundred_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 101 <= x <= 150: {_101_to_150_wrong}, sebesar: {round((_101_to_150_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 151 <= x <= 200: {_151_to_200_wrong}, sebesar: {round((_151_to_200_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 201 <= x <= 250: {_201_to_250_wrong}, sebesar: {round((_201_to_250_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 251 <= x <= 300: {_251_to_300_wrong}, sebesar: {round((_251_to_300_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks > 300: {_over_301_wrong}, sebesar: {round((_over_301_wrong/len(df) * 100), 2)} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Panjang konteks < 100 yang terprediksi benar sebesar: {(under_hundred_right+under_hundred_wrong) and round((under_hundred_right/(under_hundred_right+under_hundred_wrong) * 100), 2)} %")
        print(f"Panjang konteks 101 <= x <= 150 yang terprediksi benar sebesar: {(_101_to_150_right+_101_to_150_wrong) and round((_101_to_150_right/(_101_to_150_right+_101_to_150_wrong) * 100), 2)} %")
        print(f"Panjang konteks 151 <= x <= 200 yang terprediksi benar sebesar: {(_151_to_200_right+_151_to_200_wrong) and round((_151_to_200_right/(_151_to_200_right+_151_to_200_wrong) * 100), 2)} %")
        print(f"Panjang konteks 201 <= x <= 250 yang terprediksi benar sebesar: {(_201_to_250_right+_201_to_250_wrong) and round((_201_to_250_right/(_201_to_250_right+_201_to_250_wrong) * 100), 2)} %")
        print(f"Panjang konteks 251 <= x <= 300 yang terprediksi benar sebesar: {(_251_to_300_right+_251_to_300_wrong) and round((_251_to_300_right/(_251_to_300_right+_251_to_300_wrong) * 100), 2)} %")
        print(f"Panjang konteks > 300 yang terprediksi benar sebesar: {(_over_301_right+_over_301_wrong) and round((_over_301_right/(_over_301_right+_over_301_wrong) * 100), 2)} %")
        print()

        print("--- Bagian tentang panjang question ---")
        print(f"-- Bagian tentang panjang question yang terprediksi BENAR --")
        print(f"Panjang question 1 <= x <= 5: {q_one_to_five_right}, sebesar: {round((q_one_to_five_right/len(df) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10: {q_six_to_ten_right}, sebesar: {round((q_six_to_ten_right/len(df) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15: {q_eleven_to_fifteen_right}, sebesar: {round((q_eleven_to_fifteen_right/len(df) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20: {q_sixteen_to_twenty_right}, sebesar: {round((q_sixteen_to_twenty_right/len(df) * 100), 2)} %")
        print(f"Panjang question > 20: {q_over_twenty_right}, sebesar: {round((q_over_twenty_right/len(df) * 100), 2)} %")
        print()
        print(f"-- Bagian tentang panjang question yang terprediksi SALAH --")
        print(f"Panjang question 1 <= x <= 5: {q_one_to_five_wrong}, sebesar: {round((q_one_to_five_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10: {q_six_to_ten_wrong}, sebesar: {round((q_six_to_ten_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15: {q_eleven_to_fifteen_wrong}, sebesar: {round((q_eleven_to_fifteen_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20: {q_sixteen_to_twenty_wrong}, sebesar: {round((q_sixteen_to_twenty_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question > 20: {q_over_twenty_wrong}, sebesar: {round((q_over_twenty_wrong/len(df) * 100), 2)} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Panjang question 1 <= x <= 5 yang terprediksi benar sebesar: {(q_one_to_five_right+q_one_to_five_wrong) and round((q_one_to_five_right/(q_one_to_five_right+q_one_to_five_wrong) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10 yang terprediksi benar sebesar: {(q_six_to_ten_right+q_six_to_ten_wrong) and round((q_six_to_ten_right/(q_six_to_ten_right+q_six_to_ten_wrong) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15 yang terprediksi benar sebesar: {(q_eleven_to_fifteen_right+q_eleven_to_fifteen_wrong) and round((q_eleven_to_fifteen_right/(q_eleven_to_fifteen_right+q_eleven_to_fifteen_wrong) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20 yang terprediksi benar sebesar: {(q_sixteen_to_twenty_right+q_sixteen_to_twenty_wrong) and round((q_sixteen_to_twenty_right/(q_sixteen_to_twenty_right+q_sixteen_to_twenty_wrong) * 100), 2)} %")
        print(f"Panjang question > 20 yang terprediksi benar sebesar: {round((q_over_twenty_right+q_over_twenty_wrong) and (q_over_twenty_right/(q_over_twenty_right+q_over_twenty_wrong) * 100), 2)} %")
        print()

        print("--- Bagian tentang panjang gold answer ---")
        print(f"-- Bagian tentang panjang gold answer yang terprediksi BENAR --")
        print(f"Panjang question 1 <= x <= 5: {a_one_to_five_right}, sebesar: {round((a_one_to_five_right/len(df) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10: {a_six_to_ten_right}, sebesar: {round((a_six_to_ten_right/len(df) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15: {a_eleven_to_fifteen_right}, sebesar: {round((a_eleven_to_fifteen_right/len(df) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20: {a_sixteen_to_twenty_right}, sebesar: {round((a_sixteen_to_twenty_right/len(df) * 100), 2)} %")
        print(f"Panjang question > 20: {a_over_twenty_right}, sebesar: {round((a_over_twenty_right/len(df) * 100), 2)} %")
        print()
        print(f"-- Bagian tentang panjang gold answer yang terprediksi SALAH --")
        print(f"Panjang question 1 <= x <= 5: {a_one_to_five_wrong}, sebesar: {round((a_one_to_five_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10: {a_six_to_ten_wrong}, sebesar: {round((a_six_to_ten_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15: {a_eleven_to_fifteen_wrong}, sebesar: {round((a_eleven_to_fifteen_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20: {a_sixteen_to_twenty_wrong}, sebesar: {round((a_sixteen_to_twenty_wrong/len(df) * 100), 2)} %")
        print(f"Panjang question > 20: {a_over_twenty_wrong}, sebesar: {round((a_over_twenty_wrong/len(df) * 100), 2)} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Panjang question 1 <= x <= 5 yang terprediksi benar sebesar: {(a_one_to_five_right+a_one_to_five_wrong) and round((a_one_to_five_right/(a_one_to_five_right+a_one_to_five_wrong) * 100), 2)} %")
        print(f"Panjang question 6 <= x <= 10 yang terprediksi benar sebesar: {(a_six_to_ten_right+a_six_to_ten_wrong) and round((a_six_to_ten_right/(a_six_to_ten_right+a_six_to_ten_wrong) * 100), 2)} %")
        print(f"Panjang question 11 <= x <= 15 yang terprediksi benar sebesar: {(a_eleven_to_fifteen_right+a_eleven_to_fifteen_wrong) and round((a_eleven_to_fifteen_right/(a_eleven_to_fifteen_right+a_eleven_to_fifteen_wrong) * 100), 2)} %")
        print(f"Panjang question 16 <= x <= 20 yang terprediksi benar sebesar: {(a_sixteen_to_twenty_right+a_sixteen_to_twenty_wrong) and round((a_sixteen_to_twenty_right/(a_sixteen_to_twenty_right+a_sixteen_to_twenty_wrong) * 100), 2)} %")
        print(f"Panjang question > 20 yang terprediksi benar sebesar: {round((a_over_twenty_right+a_over_twenty_wrong) and (a_over_twenty_right/(a_over_twenty_right+a_over_twenty_wrong) * 100), 2)} %")
        print()
        
        print("--- Bagian tentang answer type ---")
        print(f"-- Bagian tentang answer type yang terprediksi BENAR --")
        print(f"Banyak answer type Person sebanyak: {num_Person_right}, sekitar {num_Person_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type NORP sebanyak: {num_NORP_right}, sekitar {num_NORP_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Facility sebanyak: {num_Facility_right}, sekitar {num_Facility_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Organization sebanyak: {num_Organization_right}, sekitar {num_Organization_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Geo-Political Entity sebanyak: {num_Geo_Political_Entity_right}, sekitar {num_Geo_Political_Entity_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Location sebanyak: {num_Location_right}, sekitar {num_Location_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Product sebanyak: {num_Product_right}, sekitar {num_Product_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Event sebanyak: {num_Event_right}, sekitar {num_Event_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Work of Art sebanyak: {num_Work_of_Art_right}, sekitar {num_Work_of_Art_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Law sebanyak: {num_Law_right}, sekitar {num_Law_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Language sebanyak: {num_Language_right}, sekitar {num_Language_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Date sebanyak: {num_Date_right}, sekitar {num_Date_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Time sebanyak: {num_Time_right}, sekitar {num_Time_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Percent sebanyak: {num_Percent_right}, sekitar {num_Percent_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Money sebanyak: {num_Money_right}, sekitar {num_Money_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Quantity sebanyak: {num_Quantity_right}, sekitar {num_Quantity_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Ordinal sebanyak: {num_Ordinal_right}, sekitar {num_Ordinal_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Cardinal sebanyak: {num_Cardinal_right}, sekitar {num_Cardinal_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type REG sebanyak: {num_REG_right}, sekitar {num_REG_right/denominator_answer_type * 100} %")
        print(f"Banyak answer type Null sebanyak: {num_null_right}, sekitar {num_null_right/denominator_answer_type * 100} %")
        print()
        print(f"-- Bagian tentang answer type yang terprediksi SALAH --")
        print(f"Banyak answer type Person sebanyak: {num_Person_wrong}, sekitar {num_Person_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type NORP sebanyak: {num_NORP_wrong}, sekitar {num_NORP_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Facility sebanyak: {num_Facility_wrong}, sekitar {num_Facility_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Organization sebanyak: {num_Organization_wrong}, sekitar {num_Organization_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Geo-Political Entity sebanyak: {num_Geo_Political_Entity_wrong}, sekitar {num_Geo_Political_Entity_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Location sebanyak: {num_Location_wrong}, sekitar {num_Location_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Product sebanyak: {num_Product_wrong}, sekitar {num_Product_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Event sebanyak: {num_Event_wrong}, sekitar {num_Event_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Work of Art sebanyak: {num_Work_of_Art_wrong}, sekitar {num_Work_of_Art_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Law sebanyak: {num_Law_wrong}, sekitar {num_Law_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Language sebanyak: {num_Language_wrong}, sekitar {num_Language_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Date sebanyak: {num_Date_wrong}, sekitar {num_Date_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Time sebanyak: {num_Time_wrong}, sekitar {num_Time_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Percent sebanyak: {num_Percent_wrong}, sekitar {num_Percent_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Money sebanyak: {num_Money_wrong}, sekitar {num_Money_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Quantity sebanyak: {num_Quantity_wrong}, sekitar {num_Quantity_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Ordinal sebanyak: {num_Ordinal_wrong}, sekitar {num_Ordinal_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Cardinal sebanyak: {num_Cardinal_wrong}, sekitar {num_Cardinal_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type REG sebanyak: {num_REG_wrong}, sekitar {num_REG_wrong/denominator_answer_type * 100} %")
        print(f"Banyak answer type Null sebanyak: {num_null_wrong}, sekitar {num_null_wrong/denominator_answer_type * 100} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Banyak answer type Person yang terprediksi benar sebesar: {(num_Person_wrong) and (round((num_Person_right/(num_Person_right+num_Person_wrong) * 100), 2))} %")
        print(f"Banyak answer type NORP yang terprediksi benar sebesar: {(num_NORP_wrong) and (round((num_NORP_right/(num_NORP_right+num_NORP_wrong) * 100), 2))} %")
        print(f"Banyak answer type Facility yang terprediksi benar sebesar: {(num_Facility_wrong) and (round((num_Facility_right/(num_Facility_right+num_Facility_wrong) * 100), 2))} %")
        print(f"Banyak answer type Organization yang terprediksi benar sebesar: {(num_Organization_wrong) and (round((num_Organization_right/(num_Organization_right+num_Organization_wrong) * 100), 2))} %")
        print(f"Banyak answer type Geo-Political Entity yang terprediksi benar sebesar: {(num_Geo_Political_Entity_wrong) and (round((num_Geo_Political_Entity_right/(num_Geo_Political_Entity_right+num_Geo_Political_Entity_wrong) * 100), 2))} %")
        print(f"Banyak answer type Location yang terprediksi benar sebesar: {(num_Location_wrong) and (round((num_Location_right/(num_Location_right+num_Location_wrong) * 100), 2))} %")
        print(f"Banyak answer type Product yang terprediksi benar sebesar: {(num_Product_wrong) and (round((num_Product_right/(num_Product_right+num_Product_wrong) * 100), 2))} %")
        print(f"Banyak answer type Event yang terprediksi benar sebesar: {(num_Event_wrong) and (round((num_Event_right/(num_Event_right+num_Event_wrong) * 100), 2))} %")
        print(f"Banyak answer type Work of Art yang terprediksi benar sebesar: {(num_Work_of_Art_wrong) and (round((num_Work_of_Art_right/(num_Work_of_Art_right+num_Work_of_Art_wrong) * 100), 2))} %")
        print(f"Banyak answer type Law yang terprediksi benar sebesar: {(num_Law_wrong) and (round((num_Law_right/(num_Law_right+num_Law_wrong) * 100), 2))} %")
        print(f"Banyak answer type Language yang terprediksi benar sebesar: {(num_Language_wrong) and (round((num_Language_right/(num_Language_right+num_Language_wrong) * 100), 2))} %")
        print(f"Banyak answer type Date yang terprediksi benar sebesar: {(num_Date_wrong) and (round((num_Date_right/(num_Date_right+num_Date_wrong) * 100), 2))} %")
        print(f"Banyak answer type Time yang terprediksi benar sebesar: {(num_Time_wrong) and (round((num_Time_right/(num_Time_right+num_Time_wrong) * 100), 2))} %")
        print(f"Banyak answer type Percent yang terprediksi benar sebesar: {(num_Percent_wrong) and (round((num_Percent_right/(num_Percent_right+num_Percent_wrong) * 100), 2))} %")
        print(f"Banyak answer type Money yang terprediksi benar sebesar: {(num_Money_wrong) and (round((num_Money_right/(num_Money_right+num_Money_wrong) * 100), 2))} %")
        print(f"Banyak answer type Quantity yang terprediksi benar sebesar: {(num_Quantity_wrong) and (round((num_Quantity_right/(num_Quantity_right+num_Quantity_wrong) * 100), 2))} %")
        print(f"Banyak answer type Ordinal yang terprediksi benar sebesar: {(num_Ordinal_wrong) and (round((num_Ordinal_right/(num_Ordinal_right+num_Ordinal_wrong) * 100), 2))} %")
        print(f"Banyak answer type Cardinal yang terprediksi benar sebesar: {(num_Cardinal_wrong) and (round((num_Cardinal_right/(num_Cardinal_right+num_Cardinal_wrong) * 100), 2))} %")
        print(f"Banyak answer type REG yang terprediksi benar sebesar: {(num_REG_wrong) and (round((num_REG_right/(num_REG_right+num_REG_wrong) * 100), 2))} %")
        print(f"Banyak answer type Null yang terprediksi benar sebesar: {(num_null_wrong) and (round((num_null_right/(num_null_right+num_null_wrong) * 100), 2))} %")
        print()
        
        print("--- Bagian tentang reasoning type ---")
        print(f"-- Bagian tentang reasoning type yang terprediksi BENAR --")
        print(f"Banyak reasoning type berjenis WM sebanyak: {num_wm_right}, sebesar: {round((num_wm_right/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis PP sebanyak: {num_pp_right}, sebesar: {round((num_pp_right/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis SSR sebanyak: {num_ssr_right}, sebesar: {round((num_ssr_right/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis MSR sebanyak: {num_msr_right}, sebesar: {round((num_msr_right/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis AoI sebanyak: {num_aoi_right}, sebesar: {round((num_aoi_right/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print()
        print(f"-- Bagian tentang reasoning type yang terprediksi SALAH --")
        print(f"Banyak reasoning type berjenis WM sebanyak: {num_wm_wrong}, sebesar: {round((num_wm_wrong/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis PP sebanyak: {num_pp_wrong}, sebesar: {round((num_pp_wrong/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis SSR sebanyak: {num_ssr_wrong}, sebesar: {round((num_ssr_wrong/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis MSR sebanyak: {num_msr_wrong}, sebesar: {round((num_msr_wrong/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print(f"Banyak reasoning type berjenis AoI sebanyak: {num_aoi_wrong}, sebesar: {round((num_aoi_wrong/NUM_REASONING_TYPE_ANNOTATED * 100), 2)} %")
        print()
        print(f"-- Presentase kebenaran --")
        print(f"Banyak reasoning type berjenis WM yang terprediksi benar sebesar: {(num_wm_wrong) and (round((num_wm_right/(num_wm_right+num_wm_wrong) * 100), 2))} %")
        print(f"Banyak reasoning type berjenis PP yang terprediksi benar sebesar: {(num_pp_wrong) and (round((num_pp_right/(num_pp_right+num_pp_wrong) * 100), 2))} %")
        print(f"Banyak reasoning type berjenis SSR yang terprediksi benar sebesar: {(num_ssr_wrong) and (round((num_ssr_right/(num_ssr_right+num_ssr_wrong) * 100), 2))} %")
        print(f"Banyak reasoning type berjenis MSR yang terprediksi benar sebesar: {(num_msr_wrong) and (round((num_msr_right/(num_msr_right+num_msr_wrong) * 100), 2))} %")
        print(f"Banyak reasoning type berjenis AoI yang terprediksi benar sebesar: {(num_aoi_wrong) and (round((num_aoi_right/(num_aoi_right+num_aoi_wrong) * 100), 2))} %")
        print()
    
    general_evaluation(qas_df)

    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/general_evaluation_results.txt', "w") as f, contextlib.redirect_stdout(f):
        general_evaluation(qas_df)
        f.close()
    
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

    print(f"Selesai fine-tuning dataset QA dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, freeze: {FREEZE}, model_sc: {MODEL_SC_NAME}, dan token: {HUB_TOKEN}")
    print("Program fine-tuning dataset QA selesai!")