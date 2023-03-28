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
parser.add_argument('-t', '--token', type=str, metavar='', required=False, help="Token Hugging Face Anda; String; choice=[all string token]; default=(TOKEN_HF_muhammadravi251001)", default="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")
parser.add_argument('-msi', '--maximum_search_iter', type=int, metavar='', required=False, help="Jumlah maximum search iter Anda; Integer; choice=[all integer]; default=2", default=2)
parser.add_argument('-tq', '--type_qas', type=str, metavar='', required=False, help="Tipe filtering QAS Anda; String; choice=[entailment only, entailment or neutral]; default=entailment or neutral", default="entailment or neutral")
parser.add_argument('-ts', '--type_smoothing', type=str, metavar='', required=False, help="Tipe smoothing hypothesis Anda; String; choice=[replace first, replace question mark, add adalah, just concat answer and question, rule based, machine generation with rule based, pure machine generation]; default=rule based", default="rule based")
args = parser.parse_args()

if __name__ == "__main__":

    base_model = ["indolem", "indonlu", "xlmr"]
    
    # Otak-atik dulu hasil HF dari Pak Aji untuk IndoNLU dan XLMR
    if (args.model_name) in base_model:
        if (args.model_name) == "indolem":
            MODEL_NAME = "afaji/fine-tuned-DatasetQAS-IDK-MRC-with-indobert-base-uncased-without-ITTL-without-freeze-LR-1e-05"
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
    MAXIMUM_SEARCH_ITER = int(args.maximum_search_iter)
    TYPE_QAS = str(args.type_qas)
    TYPE_SMOOTHING = str(args.type_smoothing)
        
    # Model-model dibawah dipilih karena memiliki akurasi terbaik dari eksperimen sebelumnya.
    if (args.model_name) == "indolem":
        MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-indobert-base-uncased"
    elif (args.model_name) == "indonlu":
        MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-indobert-large-p2"
    elif (args.model_name) == "xlmr":
        MODEL_SC_NAME = "afaji/fine-tuned-IndoNLI-Translated-with-xlm-roberta-base"
    else: MODEL_SC_NAME = str(args.model_name)

    print("Program filtering NLI mulai...")
    print(f"Mulai filtering NLI dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, model_sc: {MODEL_SC_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, dan token: {HUB_TOKEN}")

    # ## Mendefinisikan hyperparameter
    MODEL_NAME = MODEL_NAME
    MODEL_SC_NAME = MODEL_SC_NAME
    EPOCH = EPOCH
    SAMPLE = SAMPLE
    LEARNING_RATE = LEARNING_RATE
    HUB_TOKEN = HUB_TOKEN
    SEED = SEED
    BATCH_SIZE = BATCH_SIZE
    MAXIMUM_SEARCH_ITER =  MAXIMUM_SEARCH_ITER
    
    MODEL_TG_NAME = "Wikidepia/IndoT5-base-paraphrase"
    GRADIENT_ACCUMULATION = 4
    MAX_LENGTH = 400
    STRIDE = 100
    LOGGING_STEPS = 50
    WARMUP_RATIO = 0.06
    WEIGHT_DECAY = 0.01
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
    from nusacrowd import NusantaraConfigHelper
    from datetime import datetime
    from tqdm import tqdm
    from IPython.display import display
    
    from datasets import (
        load_dataset, 
        Dataset,
        DatasetDict
    )
    from transformers import (
        DataCollatorWithPadding,
        TrainingArguments,
        Trainer,
        BertForQuestionAnswering,
        AutoTokenizer,
        EarlyStoppingCallback, 
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
        df_validation = pd.DataFrame(data_qas_id['validation'])

        cols = ['context', 'question', 'answer']
        new_df_val = pd.DataFrame(columns=cols)

        for i in tqdm(range(len(df_validation['context']))):
            new_df_val = new_df_val.append({'context': df_validation["context"][i], 
                                            'question': df_validation["question"][i], 
                                            'answer': {"text": eval(df_validation["answer"][i][0])['text'], 
                                            "answer_start": eval(df_validation["answer"][i][0])['answer_start'], 
                                            "answer_end": eval(df_validation["answer"][i][0])['answer_end']}}, 
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

        train_dataset = Dataset.from_dict(new_df_train)
        validation_dataset = Dataset.from_dict(new_df_val)

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    elif (DATA_NAME == "IDK-MRC"):
        conhelps = NusantaraConfigHelper()
        data_qas_id = conhelps.filtered(lambda x: 'idk_mrc' in x.dataset_name)[0].load_dataset()

        df_train = pd.DataFrame(data_qas_id['train'])
        df_validation = pd.DataFrame(data_qas_id['validation'])

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

        train_dataset = Dataset.from_dict(new_df_train)
        validation_dataset = Dataset.from_dict(new_df_val)
        
        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    elif (DATA_NAME == "TYDI-QA-ID"):
        conhelps = NusantaraConfigHelper()
        data_qas_id = conhelps.filtered(lambda x: 'tydiqa_id' in x.dataset_name)[0].load_dataset()

        df_train = pd.DataFrame(data_qas_id['train'])
        df_validation = pd.DataFrame(data_qas_id['validation'])

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
            
        train_dataset = Dataset.from_dict(new_df_train)
        validation_dataset = Dataset.from_dict(new_df_val)

        data_qas_id = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    # ## Fungsi utilitas untuk pre-process dataset QAS
    def rindex(lst, value, operator=operator):
      return len(lst) - operator.indexOf(reversed(lst), value) - 1

    def preprocess_function_qa(examples, tokenizer, MAX_LENGTH=MAX_LENGTH, STRIDE=STRIDE, 
                               rindex=rindex, operator=operator):
        examples["question"] = [q.lstrip() for q in examples["question"]]
        examples["context"] = [c.lstrip() for c in examples["context"]]

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

    tokenized_data_qas_id = tokenized_data_qas_id.remove_columns(["offset_mapping", 
                                            "overflow_to_sample_mapping"])
    tokenized_data_qas_id.set_format("torch", columns=["input_ids", "token_type_ids"], output_all_columns=True, device=device)
    
    tokenized_data_qas_id_train = Dataset.from_dict(tokenized_data_qas_id["train"][:SAMPLE])
    tokenized_data_qas_id_validation = Dataset.from_dict(tokenized_data_qas_id["validation"][:SAMPLE])

    # # Tahapan fine-tune dataset QAS diatas model
    # ## Gunakan model Sequence Classification yang sudah pre-trained
    model_qa = BertForQuestionAnswering.from_pretrained(MODEL_NAME)
    
    desired_out_features = 2
    model_qa.qa_outputs = nn.Linear(model_qa.qa_outputs.in_features, desired_out_features)

    model_qa = model_qa.to(device)
    
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

    def compute_metrics(predict_result):
        predictions_idx = np.argmax(predict_result.predictions, axis=2)
        denominator = len(predictions_idx[0])
        label_array = np.asarray(predict_result.label_ids)
        total_correct = 0
        f1_array = []
        
        for i in range(len(predict_result.predictions[0])):
            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1
            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1

            pred_text = tokenizer.decode(tokenized_data_qas_id_validation[i]['input_ids']
                                        [start_pred_idx: end_pred_idx])
            gold_text = tokenizer.decode(tokenized_data_qas_id_validation[i]['input_ids']
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
        NAME = f'FilteringNLI-{DATA_NAME}-with-{str(MODEL_NAME)}'
    else:
        new_name = re.findall(r'.*/(.*)$', MODEL_NAME)[0]
        NAME = f'FilteringNLI-{DATA_NAME}-with-{str(new_name)}'
    
    NAME = f'{NAME}-LR-{LEARNING_RATE}'
    NAME = f'{NAME}-{TYPE_QAS}-{TYPE_SMOOTHING}'
    
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
        #hub_token=HUB_TOKEN,
        #push_to_hub=True,
        #hub_model_id=REPO_NAME,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
    )

    # ## Mulai training
    trainer_qa = Trainer(
        model=model_qa,
        args=training_args_qa,
        tokenizer=tokenizer,
    )

    # ## Simpan model
    trainer_qa.save_model(MODEL_DIR)

    # # Melakukan prediksi dari model
    predict_result = trainer_qa.predict(tokenized_data_qas_id_validation)
    os.makedirs(os.path.dirname(OUTPUT_DIR), exist_ok=True)
    with open(f'{OUTPUT_DIR}/output.txt', "w") as f:
        f.write(str(predict_result))
        f.close()
    
    metric_result_before_filtering = compute_metrics(predict_result)

    question_mark = ['siapa', 'siapakah',
                    'apa', 'apakah', 'adakah',
                    'dimana', 'dimanakah', 'darimanakah',
                    'kapan', 'kapankah',
                    'bagaimana', 'bagaimanakah',
                    'kenapa', 'mengapa',
                    'berapa', 'berapakah', 'seberapa']
    
    # # Retrieve model IndoNLI dari Hugging Face via Pipelines
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': MAX_LENGTH}

    nlp_sc = pipeline(task="text-classification", model=MODEL_SC_NAME, tokenizer=MODEL_SC_NAME, 
                    device=torch.cuda.current_device(), **tokenizer_kwargs)

    nlp_tg = pipeline(task="text2text-generation", model=MODEL_TG_NAME, tokenizer=MODEL_TG_NAME, 
                  device=torch.cuda.current_device())
    
    # # Membuat kode untuk smoothing answer dan question agar menjadi hipotesis yang natural
    def smoothing(question, pred_answer, gold_answer, type, question_mark=question_mark):
    
        if type == 'replace first':
            pred_hypothesis = question.replace('?', '')
            pred_hypothesis = pred_hypothesis.replace(question.split()[0], pred_answer)

            gold_hypothesis = question.replace('?', '')
            gold_hypothesis = gold_hypothesis.replace(question.split()[0], gold_answer)
        
        elif type == 'replace question mark':
            for i in question.split():
                if i in question_mark:
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = pred_hypothesis.replace(i, pred_answer)

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = gold_hypothesis.replace(i, gold_answer)
                else:
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = f"{pred_hypothesis.lstrip()} adalah {pred_answer}"

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = f"{gold_hypothesis.lstrip()} adalah {gold_answer}"
        
        elif type == 'add adalah':
            pred_hypothesis = question.replace('?', '')
            pred_hypothesis = pred_hypothesis.replace(question.split()[0], '')
            pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

            gold_hypothesis = question.replace('?', '')
            gold_hypothesis = gold_hypothesis.replace(question.split()[0], '')
            gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"
        
        elif type == 'just concat answer and question':
            pred_hypothesis = f"{question} {pred_answer}"         
            gold_hypothesis = f"{question} {gold_answer}"
            
        elif type == 'rule based':
            question = question.replace('kah', '')
            for j in question.split():
                if j in question_mark:
                    if j == 'siapa' or j == 'siapakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"

                    elif j == 'apa' or j == 'apakah' or j == 'adakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"

                    elif j == 'dimana' or j == 'dimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} di {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} di {gold_answer}"

                    elif j == 'darimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} dari {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} dari {gold_answer}"

                    elif j == 'kapan' or j == 'kapankah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} pada {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} pada {gold_answer}"

                    elif j == 'bagaimana' or j == 'bagaimanakah':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '')
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"

                    elif j == 'kenapa' or j == 'mengapa':
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, 'alasan').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah karena {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, 'alasan').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah karena {gold_answer}"

                    elif j == 'berapa' or j == 'berapakah' or j == 'seberapa': 
                        pred_hypothesis = question.replace('?', '')
                        pred_hypothesis = pred_hypothesis.replace(j, '').lstrip()
                        pred_hypothesis = f"{pred_hypothesis} adalah {pred_answer}"

                        gold_hypothesis = question.replace('?', '')
                        gold_hypothesis = gold_hypothesis.replace(j, '').lstrip()
                        gold_hypothesis = f"{gold_hypothesis} adalah {gold_answer}"
                else:
                    pred_hypothesis = question.replace('?', '')
                    pred_hypothesis = f"{pred_hypothesis.lstrip()} adalah {pred_answer}"

                    gold_hypothesis = question.replace('?', '')
                    gold_hypothesis = f"{gold_hypothesis.lstrip()} adalah {gold_answer}"

        elif type == 'machine generation with rule based':
            pred_hypothesis, gold_hypothesis = smoothing(question, pred_answer, gold_answer, type="rule based")
            pred_hypothesis = nlp_tg(pred_hypothesis)[0]['generated_text']
            gold_hypothesis = nlp_tg(gold_hypothesis)[0]['generated_text']
        
        elif type == 'pure machine generation':
            pred_hypothesis = f"{question} {pred_answer}"         
            gold_hypothesis = f"{question} {gold_answer}"
            
            pred_hypothesis = nlp_tg(pred_hypothesis)[0]['generated_text']
            gold_hypothesis = nlp_tg(gold_hypothesis)[0]['generated_text']
        
        return pred_hypothesis, gold_hypothesis
    
    # # Membuat kode untuk filtering answer berdasarkan label NLI: entailment (atau neutral) yang bisa menjadi hasil akhir prediksi
    def filtering_based_on_nli(predict_result, type_smoothing, type_qas, MAXIMUM_SEARCH_ITER=MAXIMUM_SEARCH_ITER):
    
    # Ekstrak dari PredictionOutput QAS
        predictions_idx = np.argsort(predict_result.predictions, axis=2)[:, :, 1 * -1]
        label_array = np.asarray(predict_result.label_ids)
        
        question_array = []
        context_array = []
        
        pred_answer_before_filtering_array = []
        pred_answer_after_filtering_array = []
        
        label_before_filtering_array = []
        label_after_filtering_array = []
        
        pred_hypothesis_before_filtering_array = []
        pred_hypothesis_after_filtering_array = []
        
        gold_answer_array = []
        gold_hypothesis_array = []
        
        # Iterasi ini ditujukan untuk retrieve answer
        for i in tqdm(range(len(predict_result.predictions[0]))):
            
            isFoundBiggest = False
            
            start_pred_idx = predictions_idx[0][i]
            end_pred_idx = predictions_idx[1][i] + 1
            
            start_gold_idx = label_array[0][i]
            end_gold_idx = label_array[1][i] + 1
            
            # Retrieve answer prediksi
            pred_answer = tokenizer.decode(tokenized_data_qas_id_validation[i]['input_ids']
                                        [start_pred_idx: end_pred_idx], skip_special_tokens=True)
            
            # Retrieve answer gold
            gold_answer = tokenizer.decode(tokenized_data_qas_id_validation[i]['input_ids']
                                        [start_gold_idx: end_gold_idx], skip_special_tokens=True)
            
            question = []
            context = []
            
            # Iterasi ini untuk retrieve question dan context index yang bersangkutan
            for j in range(len(tokenized_data_qas_id_validation[i]['token_type_ids'])):
                
                # Bila token_type_ids-nya 0, maka itu question (sesuai dengan urutan tokenisasi)
                if tokenized_data_qas_id_validation[i]['token_type_ids'][j] == 0:
                    question.append(tokenized_data_qas_id_validation[i]['input_ids'][j])
                
                # Bila token_type_ids-nya 1, maka itu context (sesuai dengan urutan tokenisasi)
                else:
                    context.append(tokenized_data_qas_id_validation[i]['input_ids'][j])
            
            question_decoded = tokenizer.decode(question, skip_special_tokens=True)
            context_decoded = tokenizer.decode(context, skip_special_tokens=True)
            pred_hypothesis, gold_hypothesis = smoothing(question_decoded, pred_answer, gold_answer, type_smoothing)

            # Cek label dari answer prediksi dan context
            predicted_label = nlp_sc({'text': context_decoded, 
                                    'text_pair': pred_hypothesis}, 
                                    **tokenizer_kwargs)
            
            pred_answer_before_filtering_array.append([pred_answer])
            pred_hypothesis_before_filtering_array.append([pred_hypothesis])
            label_before_filtering_array.append([predicted_label])
            
            # Cek label dari answer prediksi dan context, bila labelnya entailment (atau neutral), maka answernya jadi hasil akhir
            if predicted_label['label'] == 'neutral':
                if type_qas == 'entailment or neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])

            if predicted_label['label'] == 'entailment':
                if type_qas == 'entailment only' or type_qas == 'entailment or neutral':
                    question_array.append(question_decoded)
                    context_array.append(context_decoded)
                    pred_answer_after_filtering_array.append([pred_answer])
                    gold_answer_array.append(gold_answer)
                    pred_hypothesis_after_filtering_array.append([pred_hypothesis])
                    gold_hypothesis_array.append(gold_hypothesis)
                    label_after_filtering_array.append([predicted_label])
                
            # Cek label dari answer prediksi dan context, bila labelnya bukan entailment (atau neutral), 
            # -- maka masuk ke for-loop untuk iterasi ke argmax selanjutnya, dengan menggunakan argsort
            else:
                
                if predicted_label == 'neutral' and type_qas == 'entailment or neutral': continue
                
                # Bila MAXIMUM_SEARCH_ITER dibawah 2, maka continue langsung
                if MAXIMUM_SEARCH_ITER < 2: continue

                # Bila MAXIMUM_SEARCH_ITER diatas 2, maka continue langsung
                
                else:
                    # Bila bukan entailment, loop sebanyak MAXIMUM_SEARCH_ITER kali.
                    pred_answer_after_filtering_array_msi_recorded = []
                    pred_hypothesis_after_filtering_array_msi_recorded = []
                    label_after_filtering_array_msi_recorded = []
                    for index_largest in range(MAXIMUM_SEARCH_ITER - 1):
                        
                        #pred_answer_after_filtering_array_msi_recorded = []
                        #pred_hypothesis_after_filtering_array_msi_recorded = []
                        #label_after_filtering_array_msi_recorded = []

                        # Cari di index kedua, ketiga, keempat, dan seterusnya
                        predictions_idx_inside_loop = np.argsort(predict_result.predictions, 
                                                                axis=2)[:, :, (index_largest + 2) * -1]

                        start_pred_idx = predictions_idx_inside_loop[0][i]
                        end_pred_idx = predictions_idx_inside_loop[1][i] + 1

                        # Retrieve answer prediksi
                        pred_answer_inside_loop = tokenizer.decode(tokenized_data_qas_id_validation[i]['input_ids']
                                                    [start_pred_idx: end_pred_idx], skip_special_tokens=True)
                        
                        pred_hypothesis_inside_loop, gold_hypothesis = smoothing(
                            question_decoded, pred_answer_inside_loop, gold_answer, type_smoothing)
                        
                        # Cek label dari answer prediksi dan context
                        predicted_label_inside_loop = nlp_sc({'text': context_decoded, 
                                                            'text_pair': pred_hypothesis_inside_loop}
                                                            , **tokenizer_kwargs)
                        
                        pred_answer_after_filtering_array_msi_recorded.append(pred_answer_inside_loop)
                        pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_inside_loop)
                        label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)
                        
                        # Bila label-nya sudah entailment (atau neutral), maka answernya jadi hasil akhir, dan break
                        if type_qas == 'entailment only':
                            if predicted_label_inside_loop['label'] == 'entailment':
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)   
                                gold_hypothesis_array.append(gold_hypothesis)
                                
                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break
                                
                        elif type_qas == 'entailment or neutral':
                            if predicted_label_inside_loop['label'] == 'entailment' or predicted_label_inside_loop['label'] == 'neutral':
                                isFoundBiggest = True
                                question_array.append(question_decoded)
                                context_array.append(context_decoded)
                                gold_answer_array.append(gold_answer)   
                                gold_hypothesis_array.append(gold_hypothesis)
                                
                                pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                                pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                                label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
                                break

                    if isFoundBiggest == False:
                        # Bila sampai iterasi terakhir, belum entailment (atau neutral) juga, maka append saja jawaban kosong
                        
                        pred_answer_not_found_biggest = "" # Disini, jawaban kosong
                        
                        question_array.append(question_decoded)
                        context_array.append(context_decoded)
                        
                        pred_hypothesis_not_found_biggest, gold_hypothesis = smoothing(
                            question_decoded, pred_answer_not_found_biggest, gold_answer, type_smoothing)
                        
                        pred_answer_after_filtering_array_msi_recorded.append(pred_answer_not_found_biggest)
                        pred_hypothesis_after_filtering_array_msi_recorded.append(pred_hypothesis_not_found_biggest)
                        label_after_filtering_array_msi_recorded.append(predicted_label_inside_loop)
                        
                        gold_answer_array.append(gold_answer)
                        gold_hypothesis_array.append(gold_hypothesis)
                        
                        pred_answer_after_filtering_array.append(pred_answer_after_filtering_array_msi_recorded)
                        pred_hypothesis_after_filtering_array.append(pred_hypothesis_after_filtering_array_msi_recorded)
                        label_after_filtering_array.append(label_after_filtering_array_msi_recorded)
        
        # Buat DataFrame QAS
        qas_df = pd.DataFrame({'Context': context_array, 
                            'Question': question_array, 
                            
                            'Prediction Answer Before Filtering': pred_answer_before_filtering_array,
                            'Prediction Hypothesis Before Filtering': pred_hypothesis_before_filtering_array,
                            'Label Before Filtering': label_before_filtering_array,
                                    
                            'Prediction Answer After Filtering': pred_answer_after_filtering_array,
                            'Prediction Hypothesis After Filtering': pred_hypothesis_after_filtering_array,
                            'Label After Filtering': label_after_filtering_array,
                            
                            'Gold Answer': gold_answer_array,
                            'Gold Hypothesis': gold_hypothesis_array})
                            
        assert len(predict_result.predictions[0]) == len(qas_df), "Jumlah prediksi berbeda dengan jumlah evaluasi"
        
        # Return DataFrame QAS
        return qas_df
    
    filtering_result = filtering_based_on_nli(predict_result, type_smoothing=TYPE_SMOOTHING, type_qas=TYPE_QAS)

    # ## Simpan prediksi pada CSV
    filtering_result.to_csv(f'{OUTPUT_DIR}/output_df.csv')

    # # Membuat perhitungan metrik berdasarkan DataFrame, metrik yang digunakan sama persis dengan metrik sebelumnya (compute_metrics)
    def compute_metrics_from_df(df, type_qas):
    
        denominator = len(df)
        total_correct = 0
        f1_array = []
        
        true_positive_before_filtering = 0
        false_positive_before_filtering = 0
        false_negative_before_filtering = 0
        true_negative_before_filtering = 0
        
        true_positive_after_filtering = 0
        false_positive_after_filtering = 0
        false_negative_after_filtering = 0
        true_negative_after_filtering = 0

        for i in range(len(df)):
            
            pred_answer_before_filtering = df["Prediction Answer Before Filtering"][i][-1]
            pred_answer_after_filtering = df["Prediction Answer After Filtering"][i][-1]
            
            pred_label_before_filtering = df["Label Before Filtering"][i][-1]['label']
            pred_label_after_filtering = df["Label After Filtering"][i][-1]['label']
            
            gold_text = df["Gold Answer"][i]

            if pred_answer_after_filtering == gold_text:
                total_correct += 1

            f1 = compute_f1(pred=pred_answer_after_filtering, gold=gold_text)

            f1_array.append(f1)
            
            # Terprediksi dengan label yang benar, dan hasil answernya benar -> True positive
            # Terprediksi dengan label yang benar, padahal hasil answernya salah -> False positive
            # Terprediksi dengan label yang salah, padahal hasil answernya benar -> False negative
            # Terprediksi dengan label yang salah, dan hasil answernya salah -> True negative
            
            if type_qas == 'entailment only':
            
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering == 'entailment'):
                    true_positive_after_filtering += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering == 'entailment'):
                    false_positive_after_filtering += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering != 'entailment'):
                    false_negative_after_filtering += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering != 'entailment'):
                    true_negative_after_filtering += 1

                if (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'entailment'):
                    true_positive_before_filtering += 1
                elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'entailment'):
                    false_positive_before_filtering += 1
                elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering != 'entailment'):
                    false_negative_before_filtering += 1
                elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering != 'entailment'):
                    true_negative_before_filtering += 1
            
            elif type_qas == 'entailment or neutral':
            
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering == 'entailment' 
                                                                or pred_label_after_filtering == 'neutral'):
                    true_positive_after_filtering += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering == 'entailment' 
                                                                    or pred_label_after_filtering == 'neutral'):
                    false_positive_after_filtering += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering != 'entailment' 
                                                                    and pred_label_after_filtering != 'neutral'):
                    false_negative_after_filtering += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering != 'entailment' 
                                                                    and pred_label_after_filtering != 'neutral'):
                    true_negative_after_filtering += 1

                if (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'entailment' 
                                                                    or pred_label_after_filtering == 'neutral'):
                    true_positive_before_filtering += 1
                elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'entailment' 
                                                                    or pred_label_after_filtering == 'neutral'):
                    false_positive_before_filtering += 1
                elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering != 'entailment' 
                                                                    and pred_label_after_filtering != 'neutral'):
                    false_negative_before_filtering += 1
                elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering != 'entailment' 
                                                                    and pred_label_after_filtering != 'neutral'):
                    true_negative_before_filtering += 1

        exact_match = ((total_correct / denominator) * 100.0)
        final_f1 = np.mean(f1_array) * 100.0
        after_filtering_metric_array = [true_positive_after_filtering, false_positive_after_filtering, 
                            false_negative_after_filtering, true_negative_after_filtering]
        before_filtering_metric_array = [true_positive_before_filtering, false_positive_before_filtering, 
                            false_negative_before_filtering, true_negative_before_filtering]

        return {'exact_match': exact_match, 'f1': final_f1}, after_filtering_metric_array, before_filtering_metric_array
    
    metric_result_after_filtering, after_filtering_metric_array, before_filtering_metric_array = compute_metrics_from_df(
        filtering_result, type_qas=TYPE_QAS)

    # # Method-method helper untuk perhitungan metrik
    def convert_to_non_zero(number):
        if number == 0:
            number += sys.float_info.min
        return number

    def compute_f1_prec_rec_whole(metric_array):
        accuracy = (metric_array[0] + metric_array[3]) / \
            (metric_array[0] + metric_array[1] + 
            metric_array[2] + metric_array[3])
        
        precision = (metric_array[0]) / (metric_array[0] + metric_array[1])
        
        recall = (metric_array[0]) / (metric_array[0] + metric_array[2])
        
        f1 = (2 * precision * recall) / (precision + recall)
        
        return accuracy, precision, recall, f1

    def diff_verbose_metric(metric_result_before, metric_result_after, metric):
        
        percentage = round(((metric_result_after - metric_result_before) / metric_result_before) * 100, 2)
        
        if '&' in metric: vocab = "nilai"
        else: vocab = "metrik"
        
        if metric_result_before ==  metric_result_after:
            print(f"Hasil {vocab} {metric} sebelum filtering NLI SAMA DENGAN metrik setelah filtering NLI")
        elif metric_result_before <  metric_result_after:
            print(f"Hasil {vocab} {metric} setelah filtering NLI mengalami KENAIKAN sebesar: {percentage} %")
        elif metric_result_before >  metric_result_after:
            print(f"Hasil {vocab} {metric} setelah filtering NLI mengalami PENURUNAN sebesar: {-1 * percentage} %")
        
        return percentage
    
    # # Membandingkan metrik sebelum filtering NLI dan setelah filtering NLI
    def compare_metrics(metrics_before, metrics_after, 
                    after_filtering_metric_array=after_filtering_metric_array, 
                    before_filtering_metric_array=before_filtering_metric_array):
    
        em_before = metrics_before['exact_match']
        f1_before = metrics_before['f1']

        print("~ METRIK PER TOKEN ~")
        print(f"Skor Exact Match sebelum filtering NLI: {em_before}")
        print(f"Skor F1 sebelum filtering NLI: {f1_before}")
        print()

        em_after = metrics_after['exact_match']
        f1_after = metrics_after['f1']

        print(f"Skor Exact Match setelah filtering NLI: {em_after}")
        print(f"Skor F1 setelah filtering NLI: {f1_after}")
        print()

        em_before = convert_to_non_zero(em_before)
        f1_before = convert_to_non_zero(f1_before)

        em_after = convert_to_non_zero(em_after)
        f1_after = convert_to_non_zero(f1_after)

        print("~ METRIK DENGAN PARAMETER NLI ~")
        print(f"[BEFORE FILTERING] Jawaban benar & label NLI yang sesuai: {before_filtering_metric_array[0]}")
        print(f"[BEFORE FILTERING] Jawaban TIDAK benar & label NLI yang sesuai: {before_filtering_metric_array[1]}")
        print(f"[BEFORE FILTERING] Jawaban benar & label NLI yang TIDAK sesuai: {before_filtering_metric_array[2]}")
        print(f"[BEFORE FILTERING] Jawaban TIDAK benar & label NLI yang TIDAK sesuai: {before_filtering_metric_array[3]}")
        print()

        print(f"[AFTER FILTERING] Jawaban benar & label NLI yang sesuai: {after_filtering_metric_array[0]}")
        print(f"[AFTER FILTERING] Jawaban TIDAK benar & label NLI yang sesuai: {after_filtering_metric_array[1]}")
        print(f"[AFTER FILTERING] Jawaban benar & label NLI yang TIDAK sesuai: {after_filtering_metric_array[2]}")
        print(f"[AFTER FILTERING] Jawaban TIDAK benar & label NLI yang TIDAK sesuai: {after_filtering_metric_array[3]}")
        print()

        print("Metrik di atas, bisa direpresentasikan menjadi:")

        acc_before_whole, prec_before_whole, rec_before_whole, f1_before_whole = compute_f1_prec_rec_whole(
            before_filtering_metric_array)
        acc_after_whole, prec_after_whole, rec_after_whole, f1_after_whole = compute_f1_prec_rec_whole(
            after_filtering_metric_array)

        print(f"[BEFORE FILTERING] Akurasi: {acc_before_whole}")
        print(f"[BEFORE FILTERING] Precision: {prec_before_whole}")
        print(f"[BEFORE FILTERING] Recall: {rec_before_whole}")
        print(f"[BEFORE FILTERING] F1: {f1_before_whole}")
        print()

        print(f"[AFTER FILTERING] Akurasi: {acc_after_whole}")
        print(f"[AFTER FILTERING] Precision: {prec_after_whole}")
        print(f"[AFTER FILTERING] Recall: {rec_after_whole}")
        print(f"[AFTER FILTERING] F1: {f1_after_whole}")
        print()

        print("--- Persentase perubahan hasil metrik ---")
        print("~ METRIK PER TOKEN ~")
        diff_verbose_metric(em_before, em_after, "Exact Match")
        diff_verbose_metric(f1_before, f1_after, "F1")
        print()

        print("~ METRIK DENGAN PARAMETER NLI ~")
        diff_verbose_metric(before_filtering_metric_array[0], after_filtering_metric_array[0], 
                            "Jawaban benar & label NLI yang sesuai")
        diff_verbose_metric(before_filtering_metric_array[1], after_filtering_metric_array[1], 
                            "Jawaban TIDAK benar & label NLI yang sesuai")
        diff_verbose_metric(before_filtering_metric_array[2], after_filtering_metric_array[2], 
                            "Jawaban benar & label NLI yang TIDAK sesuai")
        diff_verbose_metric(before_filtering_metric_array[3], after_filtering_metric_array[3], 
                            "Jawaban TIDAK benar & label NLI yang TIDAK sesuai")
        print()

        print("Metrik di atas, bisa direpresentasikan menjadi:")
        diff_verbose_metric(acc_before_whole, acc_after_whole, "Akurasi")
        diff_verbose_metric(prec_before_whole, prec_after_whole, "Precision")
        diff_verbose_metric(rec_before_whole, rec_after_whole, "Recall")
        diff_verbose_metric(f1_before_whole, f1_after_whole, "F1")
        print()

    compare_metrics(metric_result_before_filtering, metric_result_after_filtering)

    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/metric_comparison_results.txt', "w") as f, contextlib.redirect_stdout(f):
        compare_metrics(metric_result_before_filtering, metric_result_after_filtering)
        f.close()

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
        
        # Cek question type, yang berhasil berapa, yang gagal berapa?
        for i in range(len(df)):
            
            pred_answer_after_filtering = df["Prediction Answer After Filtering"][i][-1]       
            gold_text = df["Gold Answer"][i]
            current_question = df["Question"][i].split()
            
            if (pred_answer_after_filtering == gold_text):
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
            
            elif (pred_answer_after_filtering != gold_text):
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
        
        assert len(df) == num_apa_right+num_dimana_right+num_kapan_right+num_siapa_right+\
                            num_bagaimana_right+num_kenapa_right+num_berapa_right+num_others_right+\
                            num_apa_wrong+num_dimana_wrong+num_kapan_wrong+num_siapa_wrong+\
                            num_bagaimana_wrong+num_kenapa_wrong+num_berapa_wrong+num_others_wrong
        
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
        
        # Cek panjang passage, ada hubungan dengan berhasil/gagal?
        for i in range(len(df)):
            
            pred_answer_after_filtering = df["Prediction Answer After Filtering"][i][-1]       
            gold_text = df["Gold Answer"][i]
            len_current_passage = len(df["Context"][i].split())
            
            if (pred_answer_after_filtering == gold_text):
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
                    
            elif (pred_answer_after_filtering != gold_text):
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
                    
        assert len(df) == under_hundred_right+_101_to_150_right+_151_to_200_right+_201_to_250_right+\
                            _251_to_300_right+_over_301_right+\
                            under_hundred_wrong+_101_to_150_wrong+_151_to_200_wrong+_201_to_250_wrong+\
                            _251_to_300_wrong+_over_301_wrong
        
        # Ambil berapa contoh yang gagal, coba pelajari reasoning type-nya.
        new_df = df.sample(n=15, random_state=42)
            
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
        print(f"Panjang konteks < 100: {under_hundred_right}, sebesar: {round((under_hundred_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 101 <= x <= 150: {_101_to_150_wrong}, sebesar: {round((_101_to_150_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 151 <= x <= 200: {_151_to_200_wrong}, sebesar: {round((_151_to_200_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 201 <= x <= 250: {_201_to_250_wrong}, sebesar: {round((_201_to_250_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks 251 <= x <= 300: {_251_to_300_wrong}, sebesar: {round((_251_to_300_wrong/len(df) * 100), 2)} %")
        print(f"Panjang konteks > 300: {_over_301_wrong}, sebesar: {round((_over_301_wrong/len(df) * 100), 2)} %")
        print()
        
        print("--- Bagian untuk analisis REASONING TYPE ---")
        display(new_df)
        return new_df
    
    reasoning_type_df = general_evaluation(filtering_result)

    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/general_evaluation_results.txt', "w") as f, contextlib.redirect_stdout(f):
        general_evaluation(filtering_result)
        f.close()

    reasoning_type_df.to_csv(f'{OUTPUT_DIR}/sample_df_for_reasoning_type.csv')

    ## Breakdown evaluasi, melakukan evaluasi lebih dalam lagi
    def breakdown_evaluation(df, TYPE_QAS):
    
        if TYPE_QAS == 'entailment only': compatible_label = ['entailment']
        elif TYPE_QAS == 'entailment or neutral': compatible_label = ['entailment', 'neutral']
        
        exist_true_answer_label_entailment = 0
        exist_true_answer_label_neutral = 0
        exist_true_answer_label_contradiction = 0
        
        exist_false_answer_label_entailment = 0
        exist_false_answer_label_neutral = 0
        exist_false_answer_label_contradiction = 0
        
        no_exist_true_answer_label_entailment = 0
        no_exist_true_answer_label_neutral = 0
        no_exist_true_answer_label_contradiction = 0
        
        no_exist_false_answer_label_entailment = 0
        no_exist_false_answer_label_neutral = 0
        no_exist_false_answer_label_contradiction = 0
        
        filtered_in_right_answer_to_filtered_in_right_answer = 0
        filtered_in_right_answer_to_filtered_in_wrong_answer = 0
        filtered_in_right_answer_to_filtered_out_right_answer = 0
        filtered_in_right_answer_to_filtered_out_wrong_answer = 0
        
        filtered_in_wrong_answer_to_filtered_in_right_answer = 0
        filtered_in_wrong_answer_to_filtered_in_wrong_answer = 0
        filtered_in_wrong_answer_to_filtered_out_right_answer = 0
        filtered_in_wrong_answer_to_filtered_out_wrong_answer = 0
        
        filtered_out_right_answer_to_filtered_in_right_answer = 0
        filtered_out_right_answer_to_filtered_in_wrong_answer = 0
        filtered_out_right_answer_to_filtered_out_right_answer = 0
        filtered_out_right_answer_to_filtered_out_wrong_answer = 0
        
        filtered_out_wrong_answer_to_filtered_in_right_answer = 0
        filtered_out_wrong_answer_to_filtered_in_wrong_answer = 0
        filtered_out_wrong_answer_to_filtered_out_right_answer = 0
        filtered_out_wrong_answer_to_filtered_out_wrong_answer = 0
        
        filtered_in_right_answer_to_filtered_in_right_answer_unanswered = 0
        filtered_in_right_answer_to_filtered_in_wrong_answer_unanswered = 0
        filtered_in_right_answer_to_filtered_out_right_answer_unanswered = 0
        filtered_in_right_answer_to_filtered_out_wrong_answer_unanswered = 0
        
        filtered_in_wrong_answer_to_filtered_in_right_answer_unanswered = 0
        filtered_in_wrong_answer_to_filtered_in_wrong_answer_unanswered = 0
        filtered_in_wrong_answer_to_filtered_out_right_answer_unanswered = 0
        filtered_in_wrong_answer_to_filtered_out_wrong_answer_unanswered = 0
        
        filtered_out_right_answer_to_filtered_in_right_answer_unanswered = 0
        filtered_out_right_answer_to_filtered_in_wrong_answer_unanswered = 0
        filtered_out_right_answer_to_filtered_out_right_answer_unanswered = 0
        filtered_out_right_answer_to_filtered_out_wrong_answer_unanswered = 0
        
        filtered_out_wrong_answer_to_filtered_in_right_answer_unanswered = 0
        filtered_out_wrong_answer_to_filtered_in_wrong_answer_unanswered = 0
        filtered_out_wrong_answer_to_filtered_out_right_answer_unanswered = 0
        filtered_out_wrong_answer_to_filtered_out_wrong_answer_unanswered = 0
        
        filtered_score_labels_before_filtering = []
        filtered_score_labels_after_filtering = []
        
        for i in range(len(df)):
            
            pred_answer_before_filtering = df["Prediction Answer Before Filtering"][i][-1]
            pred_answer_after_filtering = df["Prediction Answer After Filtering"][i][-1]
            
            pred_label_before_filtering = df["Label Before Filtering"][i][-1]['label']
            pred_label_after_filtering = df["Label After Filtering"][i][-1]['label']
            
            pred_prob_dist_before_filtering = df["Label Before Filtering"][i][-1]['score']
            pred_prob_dist_after_filtering = df["Label After Filtering"][i][-1]['score']
            
            gold_text = df["Gold Answer"][i]

            # Bagian untuk jawaban sebelum filtering SAMA DENGAN ground truth
            
            if (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'entailment') \
                    and (pred_answer_before_filtering != ""): 
                exist_true_answer_label_entailment += 1
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_right_answer_to_filtered_in_right_answer += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_right_answer_to_filtered_in_wrong_answer += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_right_answer_to_filtered_out_right_answer += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_right_answer_to_filtered_out_wrong_answer += 1
            
            elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'neutral') \
                    and (pred_answer_before_filtering != ""): 
                exist_true_answer_label_neutral += 1
                
                if (TYPE_QAS == 'entailment only'):
                    
                    filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                    
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_right_answer_to_filtered_in_right_answer += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_right_answer_to_filtered_in_wrong_answer += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_right_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_right_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                
                elif (TYPE_QAS == 'entailment or neutral'):
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_right_answer_to_filtered_in_right_answer += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_right_answer_to_filtered_in_wrong_answer += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_right_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_right_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    
            elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'contradiction') \
                    and (pred_answer_before_filtering != ""): 
                exist_true_answer_label_contradiction += 1
                
                filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_right_answer_to_filtered_in_right_answer += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_right_answer_to_filtered_in_wrong_answer += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_right_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_right_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            # Bagian untuk jawaban sebelum filtering BERBEDA DENGAN ground truth
            
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'entailment') \
                    and (pred_answer_before_filtering != ""):
                exist_false_answer_label_entailment += 1
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_in_right_answer += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_in_wrong_answer += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'neutral') \
                    and (pred_answer_before_filtering != ""):
                exist_false_answer_label_neutral += 1
                
                if (TYPE_QAS == 'entailment only'):
                    
                    filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                    
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_in_right_answer += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_in_wrong_answer += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                
                elif (TYPE_QAS == 'entailment or neutral'):
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_in_right_answer += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_in_wrong_answer += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                        
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'contradiction') \
                    and (pred_answer_before_filtering != ""):
                exist_false_answer_label_contradiction += 1
                
                filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_in_right_answer += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_in_wrong_answer += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_out_right_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_out_wrong_answer += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            # Bagian untuk jawaban sebelum filtering SAMA DENGAN ground truth (unanswered)
            
            elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'entailment') \
                    and (pred_answer_before_filtering == ""): 
                no_exist_true_answer_label_entailment += 1
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_right_answer_to_filtered_in_right_answer_unanswered += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_right_answer_to_filtered_in_wrong_answer_unanswered += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_right_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_right_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'neutral') \
                    and (pred_answer_before_filtering == ""): 
                no_exist_true_answer_label_neutral += 1
                
                if (TYPE_QAS == 'entailment only'):
                    
                    filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                    
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_right_answer_to_filtered_in_right_answer_unanswered += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_right_answer_to_filtered_in_wrong_answer_unanswered += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_right_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_right_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                
                elif (TYPE_QAS == 'entailment or neutral'):
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_right_answer_to_filtered_in_right_answer_unanswered += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_right_answer_to_filtered_in_wrong_answer_unanswered += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_right_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_right_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    
            elif (pred_answer_before_filtering == gold_text) and (pred_label_before_filtering == 'contradiction') \
                    and (pred_answer_before_filtering == ""): 
                no_exist_true_answer_label_contradiction += 1
                
                filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_right_answer_to_filtered_in_right_answer_unanswered += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_right_answer_to_filtered_in_wrong_answer_unanswered += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_right_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_right_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            # Bagian untuk jawaban sebelum filtering BERBEDA DENGAN ground truth (unanswered)
            
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'entailment') \
                    and (pred_answer_before_filtering == ""):
                no_exist_false_answer_label_entailment += 1
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_in_right_answer_unanswered += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_in_wrong_answer_unanswered += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_in_wrong_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
            
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'neutral') \
                    and (pred_answer_before_filtering == ""):
                no_exist_false_answer_label_neutral += 1
                
                if (TYPE_QAS == 'entailment only'):
                    
                    filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                    
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_in_right_answer_unanswered += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_in_wrong_answer_unanswered += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_out_wrong_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                
                elif (TYPE_QAS == 'entailment or neutral'):
                    if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_in_right_answer_unanswered += 1
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_in_wrong_answer_unanswered += 1
                    elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                    elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                            : filtered_in_wrong_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                        
            elif (pred_answer_before_filtering != gold_text) and (pred_label_before_filtering == 'contradiction') \
                    and (pred_answer_before_filtering == ""):
                no_exist_false_answer_label_contradiction += 1
                
                filtered_score_labels_before_filtering.append(pred_prob_dist_before_filtering)
                
                if (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_in_right_answer_unanswered += 1
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_in_wrong_answer_unanswered += 1
                elif (pred_answer_after_filtering == gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_out_right_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                elif (pred_answer_after_filtering != gold_text) and (pred_label_after_filtering not in compatible_label) \
                        : filtered_out_wrong_answer_to_filtered_out_wrong_answer_unanswered += 1; filtered_score_labels_after_filtering.append(pred_prob_dist_after_filtering)
                
            #print(f"Pred answer before filtering: {pred_answer_before_filtering}")
            #print(f"Pred answer after filtering: {pred_answer_after_filtering}")
            #print(f"Gold answer: {gold_text}")
            #print()
        
        print(f"--- Bagian ini hanya memperhatikan sebelum filtering ---")
        print(f"Jawaban benar (answer exist) entailment: {exist_true_answer_label_entailment}, sebesar: {round(exist_true_answer_label_entailment/len(df) * 100, 2)} %")
        print(f"Jawaban benar (answer exist) neutral: {exist_true_answer_label_neutral}, sebesar: {round(exist_true_answer_label_neutral/len(df) * 100, 2)} %")
        print(f"Jawaban benar (answer exist) contradiction: {exist_true_answer_label_contradiction}, sebesar: {round(exist_true_answer_label_contradiction/len(df) * 100, 2)} %")
        print()
        print(f"Jawaban salah (answer exist) entailment: {exist_false_answer_label_entailment}, sebesar: {round(exist_false_answer_label_entailment/len(df) * 100, 2)} %")
        print(f"Jawaban salah (answer exist) neutral: {exist_false_answer_label_neutral}, sebesar: {round(exist_false_answer_label_neutral/len(df) * 100, 2)} %")
        print(f"Jawaban salah (answer exist) contradiction: {exist_false_answer_label_contradiction}, sebesar: {round(exist_false_answer_label_contradiction/len(df) * 100, 2)} %")
        print()
        print(f"Jawaban benar (answer DO NOT exist) entailment: {no_exist_true_answer_label_entailment}, sebesar: {round(no_exist_true_answer_label_entailment/len(df) * 100, 2)} %")
        print(f"Jawaban benar (answer DO NOT exist) neutral: {no_exist_true_answer_label_neutral}, sebesar: {round(no_exist_true_answer_label_neutral/len(df) * 100, 2)} %")
        print(f"Jawaban benar (answer DO NOT exist) contradiction: {no_exist_true_answer_label_contradiction}, sebesar: {round(no_exist_true_answer_label_contradiction/len(df) * 100, 2)} %")
        print()
        print(f"Jawaban salah (answer DO NOT exist) entailment: {no_exist_false_answer_label_entailment}, sebesar: {round(no_exist_false_answer_label_entailment/len(df) * 100, 2)} %")
        print(f"Jawaban salah (answer DO NOT exist) neutral: {no_exist_false_answer_label_neutral}, sebesar: {round(no_exist_false_answer_label_neutral/len(df) * 100, 2)} %")
        print(f"Jawaban salah (answer DO NOT exist) contradiction: {no_exist_false_answer_label_contradiction}, sebesar: {round(no_exist_false_answer_label_contradiction/len(df) * 100, 2)} %")
        print()
        
        print(f"--- Bagian ini memperhatikan sebelum filtering dan setelah filtering ---")
        
        print(f"Banyaknya data sebelum filtering: BENAR & LOLOS, setelah filtering: BENAR & LOLOS: {filtered_in_right_answer_to_filtered_in_right_answer}, sebesar: {round(filtered_in_right_answer_to_filtered_in_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & LOLOS, setelah filtering: SALAH & LOLOS: {filtered_in_right_answer_to_filtered_in_wrong_answer}, sebesar: {round(filtered_in_right_answer_to_filtered_in_wrong_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & LOLOS, setelah filtering: BENAR & TERFILTER: {filtered_in_right_answer_to_filtered_out_right_answer}, sebesar: {round(filtered_in_right_answer_to_filtered_out_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & LOLOS, setelah filtering: SALAH & TERFILTER: {filtered_in_right_answer_to_filtered_out_wrong_answer}, sebesar: {round(filtered_in_right_answer_to_filtered_out_wrong_answer/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering: SALAH & LOLOS, setelah filtering: BENAR & LOLOS: {filtered_in_wrong_answer_to_filtered_in_right_answer}, sebesar: {round(filtered_in_wrong_answer_to_filtered_in_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & LOLOS, setelah filtering: SALAH & LOLOS: {filtered_in_wrong_answer_to_filtered_in_wrong_answer}, sebesar: {round(filtered_in_wrong_answer_to_filtered_in_wrong_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & LOLOS, setelah filtering: BENAR & TERFILTER: {filtered_in_wrong_answer_to_filtered_out_right_answer}, sebesar: {round(filtered_in_wrong_answer_to_filtered_out_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & LOLOS, setelah filtering: SALAH & TERFILTER: {filtered_in_wrong_answer_to_filtered_out_wrong_answer}, sebesar: {round(filtered_in_wrong_answer_to_filtered_out_wrong_answer/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering: BENAR & TERFILTER, setelah filtering: BENAR & LOLOS: {filtered_out_right_answer_to_filtered_in_right_answer}, sebesar: {round(filtered_out_right_answer_to_filtered_in_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & TERFILTER, setelah filtering: SALAH & LOLOS: {filtered_out_right_answer_to_filtered_in_wrong_answer}, sebesar: {round(filtered_out_right_answer_to_filtered_in_wrong_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & TERFILTER, setelah filtering: BENAR & TERFILTER: {filtered_out_right_answer_to_filtered_out_right_answer}, sebesar: {round(filtered_out_right_answer_to_filtered_out_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: BENAR & TERFILTER, setelah filtering: SALAH & TERFILTER: {filtered_out_right_answer_to_filtered_out_wrong_answer}, sebesar: {round(filtered_out_right_answer_to_filtered_out_wrong_answer/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering: SALAH & TERFILTER, setelah filtering: BENAR & LOLOS: {filtered_out_wrong_answer_to_filtered_in_right_answer}, sebesar: {round(filtered_out_wrong_answer_to_filtered_in_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & TERFILTER, setelah filtering: SALAH & LOLOS: {filtered_out_wrong_answer_to_filtered_in_wrong_answer}, sebesar: {round(filtered_out_wrong_answer_to_filtered_in_wrong_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & TERFILTER, setelah filtering: BENAR & TERFILTER: {filtered_out_wrong_answer_to_filtered_out_right_answer}, sebesar: {round(filtered_out_wrong_answer_to_filtered_out_right_answer/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering: SALAH & TERFILTER, setelah filtering: SALAH & TERFILTER: {filtered_out_wrong_answer_to_filtered_out_wrong_answer}, sebesar: {round(filtered_out_wrong_answer_to_filtered_out_wrong_answer/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & LOLOS, setelah filtering: BENAR & LOLOS: {filtered_in_right_answer_to_filtered_in_right_answer_unanswered}, sebesar: {round(filtered_in_right_answer_to_filtered_in_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & LOLOS, setelah filtering: SALAH & LOLOS: {filtered_in_right_answer_to_filtered_in_wrong_answer_unanswered}, sebesar: {round(filtered_in_right_answer_to_filtered_in_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & LOLOS, setelah filtering: BENAR & TERFILTER: {filtered_in_right_answer_to_filtered_out_right_answer_unanswered}, sebesar: {round(filtered_in_right_answer_to_filtered_out_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & LOLOS, setelah filtering: SALAH & TERFILTER: {filtered_in_right_answer_to_filtered_out_wrong_answer_unanswered}, sebesar: {round(filtered_in_right_answer_to_filtered_out_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & LOLOS, setelah filtering: BENAR & LOLOS: {filtered_in_wrong_answer_to_filtered_in_right_answer_unanswered}, sebesar: {round(filtered_in_wrong_answer_to_filtered_in_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & LOLOS, setelah filtering: SALAH & LOLOS: {filtered_in_wrong_answer_to_filtered_in_wrong_answer_unanswered}, sebesar: {round(filtered_in_wrong_answer_to_filtered_in_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & LOLOS, setelah filtering: BENAR & TERFILTER: {filtered_in_wrong_answer_to_filtered_out_right_answer_unanswered}, sebesar: {round(filtered_in_wrong_answer_to_filtered_out_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & LOLOS, setelah filtering: SALAH & TERFILTER: {filtered_in_wrong_answer_to_filtered_out_wrong_answer_unanswered}, sebesar: {round(filtered_in_wrong_answer_to_filtered_out_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & TERFILTER, setelah filtering: BENAR & LOLOS: {filtered_out_right_answer_to_filtered_in_right_answer_unanswered}, sebesar: {round(filtered_out_right_answer_to_filtered_in_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & TERFILTER, setelah filtering: SALAH & LOLOS: {filtered_out_right_answer_to_filtered_in_wrong_answer_unanswered}, sebesar: {round(filtered_out_right_answer_to_filtered_in_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & TERFILTER, setelah filtering: BENAR & TERFILTER: {filtered_out_right_answer_to_filtered_out_right_answer_unanswered}, sebesar: {round(filtered_out_right_answer_to_filtered_out_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): BENAR & TERFILTER, setelah filtering: SALAH & TERFILTER: {filtered_out_right_answer_to_filtered_out_wrong_answer_unanswered}, sebesar: {round(filtered_out_right_answer_to_filtered_out_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print()
        
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & TERFILTER, setelah filtering: BENAR & LOLOS: {filtered_out_wrong_answer_to_filtered_in_right_answer_unanswered}, sebesar: {round(filtered_out_wrong_answer_to_filtered_in_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & TERFILTER, setelah filtering: SALAH & LOLOS: {filtered_out_wrong_answer_to_filtered_in_wrong_answer_unanswered}, sebesar: {round(filtered_out_wrong_answer_to_filtered_in_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & TERFILTER, setelah filtering: BENAR & TERFILTER: {filtered_out_wrong_answer_to_filtered_out_right_answer_unanswered}, sebesar: {round(filtered_out_wrong_answer_to_filtered_out_right_answer_unanswered/len(df) * 100, 2)} %")
        print(f"Banyaknya data sebelum filtering (unanswered): SALAH & TERFILTER, setelah filtering: SALAH & TERFILTER: {filtered_out_wrong_answer_to_filtered_out_wrong_answer_unanswered}, sebesar: {round(filtered_out_wrong_answer_to_filtered_out_wrong_answer_unanswered/len(df) * 100, 2)} %")
        print()
        
        print(f"Rerata skor yang membuat data menjadi TERFILTER sebelum iterasi MSI: {np.mean(filtered_score_labels_before_filtering)}")
        print(f"Rerata skor yang membuat data menjadi TERFILTER setelah iterasi MSI: {np.mean(filtered_score_labels_after_filtering)}")
        print(f"Total prediksi jawaban: {len(df)}")
        print()
        
        assert len(df) == exist_true_answer_label_entailment+exist_true_answer_label_neutral+exist_true_answer_label_contradiction+\
                exist_false_answer_label_entailment+exist_false_answer_label_neutral+exist_false_answer_label_contradiction+\
                no_exist_true_answer_label_entailment+no_exist_true_answer_label_neutral+no_exist_true_answer_label_contradiction+\
                no_exist_false_answer_label_entailment+no_exist_false_answer_label_neutral+no_exist_false_answer_label_contradiction

    breakdown_evaluation(filtering_result, TYPE_QAS=TYPE_QAS)

    os.makedirs(os.path.dirname(METRIC_RESULT_DIR), exist_ok=True)
    with open(f'{METRIC_RESULT_DIR}/breakdown_evaluation_results.txt', "w") as f, contextlib.redirect_stdout(f):
        breakdown_evaluation(filtering_result, TYPE_QAS=TYPE_QAS)
        f.close()

    print(f"Selesai filtering NLI dengan model: {MODEL_NAME} dan data: {DATA_NAME}, dengan epoch: {EPOCH}, sample: {SAMPLE}, LR: {LEARNING_RATE}, seed: {SEED}, batch_size: {BATCH_SIZE}, model_sc: {MODEL_SC_NAME}, tq: {TYPE_QAS}, ts: {TYPE_SMOOTHING}, msi: {MAXIMUM_SEARCH_ITER}, dan token: {HUB_TOKEN}")
    print("Program filtering NLI selesai!")