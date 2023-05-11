# Experiments

## Installation and Setup

First, use virtual environment and clone this repository:
```
python -m venv qas-with-indonli-env
qas-with-indonli-env\scripts\activate
git clone https://github.com/muhammadravi251001/qas-with-indonli.git
```

After that, run experiment and install all the requirements under the current directory:
```
cd qas-with-indonli
cd experiments
pip install -r requirements.txt
```

If you not yet install git-lfs, please install. You can install git-lfs like this:
```
git lfs install
```
Or, like this:
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install -y git-lfs
git lfs install
```

## Running experiments for training IndoNLI

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_training_indonli.py -h
```

To run this training IndoNLI experiments, you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size, etc} if you don't want using the default value provided:
```
python main_training_indonli.py -m indolem -d basic -e 10 -sa max
python main_training_indonli.py -m indolem -d translated -e 10 -sa max
python main_training_indonli.py -m indolem -d augmented -e 10 -sa max

python main_training_indonli.py -m indonlu -d basic -e 10 -sa max
python main_training_indonli.py -m indonlu -d translated -e 10 -sa max
python main_training_indonli.py -m indonlu -d augmented -e 10 -sa max

python main_training_indonli.py -m xlmr -d basic -e 10 -sa max
python main_training_indonli.py -m xlmr -d translated -e 10 -sa max
python main_training_indonli.py -m xlmr -d augmented -e 10 -sa max
```

## Running experiments for fine-tuning dataset QAS

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_fine_tuning_qas_dataset.py -h
```

To run this fine-tuning QAS datasets experiments WITHOUT Intermediate Task Transfer Learning (ITTL), you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size, etc} if you don't want using the default value provided:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqaid -e 16 -sa max -wi False

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqaid -e 16 -sa max -wi False

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max -wi False
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqaid -e 16 -sa max -wi False
```

It should be understood that when you use the `-wi False` flag it means that you are fine-tuning the QAS dataset with the baseline flow, usually this is done as a reference whether the performance of a QAS is increasing or decreasing based on the `-wi False` baseline flow. By default, if you do not config `-wi` flag, is set to be False.

And then, to run this fine-tuning QAS datasets experiments WITH Intermediate Task Transfer Learning (ITTL), you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size, etc} if you don't want using the default value provided:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max -wi True
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max  -wi True
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqaid -e 16 -sa max -wi True

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max -wi True
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max -wi True
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqaid -e 16 -sa max -wi True

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max -wi True
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max -wi True
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqaid -e 16 -sa max -wi True
```

Or, if you want to experiment with freezing your BERT layer (the default is without freezing BERT layer). You can do this:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max -wi True -fr True
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max  -wi True -fr True
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqaid -e 16 -sa max -wi True -fr True

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max -wi True -fr True
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max -wi True -fr True
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqaid -e 16 -sa max -wi True -fr True

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max -wi True -fr True
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max -wi True -fr True
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqaid -e 16 -sa max -wi True -fr True
```

With use of `-msc {YOUR_MODEL_CHOICE}` flag, that means you doing Intermediate Task Transfer Learning (ITTL). It's like you do fine-tuning twice, first with the Sequence Classification task with IndoNLI dataset and the second with the Question Answering task with the QAS dataset that you have chosen yourself. You can not choose your intermediate task model freely without having to match the `{MODEL_NAME}` of Question Answering task that you previously chose. You must have the same `{MODEL_NAME}` as the Question Answering task, however, we recommend the models with the best results on the Hugging Face [@afaji](https://huggingface.co/afaji) account.

## Running experiments for filtering QAS based on NLI

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python filtering_nli_experiment.py -h
```

To run this filtering QAS based on NLI experiments, you just only do this, you optionally need to passing arguments to if you don't want using the default value provided:
```
python filtering_nli_experiment.py -m indolem -d squadid 
python filtering_nli_experiment.py -m indolem -d idkmrc
python filtering_nli_experiment.py -m indolem -d tydiqaid

python filtering_nli_experiment.py -m indonlu -d squadid
python filtering_nli_experiment.py -m indonlu -d idkmrc
python filtering_nli_experiment.py -m indonlu -d tydiqaid

python filtering_nli_experiment.py -m xlmr -d squadid
python filtering_nli_experiment.py -m xlmr -d idkmrc
python filtering_nli_experiment.py -m xlmr -d tydiqaid
```

However, you can try some of these flags for your experiments, namely: `-tq`, `-ts`, `-msi`, `-va`, and `-th`.

`-tq` stands for `TYPE_QAS`, in this flag, you can experimenting how should your QAS system filtering based on NLI, whether you choose `entailment_only` or `entailment_or_neutral`. 

`-ts` stands for `TYPE_SMOOTHING`, in this flag, you can experimenting how should your smoothing system, to get your hypothesis as natural as possible from your answer and question from your QAS dataset, you can choose: `replace_first, replace_question_word, add_adalah, just_concat answer_and_question, rule_based, machine_generation_with_rule_based, pure_machine_generation, machine_generation_with_translation`. 

`-msi` stands for `MAXIMUM_SEARCH_ITER`, in this flag, you can experimenting how much your QAS system search for entailment label (or neutral label, it depends on your choice in `TYPE_QAS`) from your QAS dataset, you can choose any integer to fill in this flag.

`-va` stands for `VARIATION`, in this flag, you can experimenting what variation your QAS system from your QAS dataset, you can choose 0, 1, 2 to fill in this flag. Variation 1 means you use the discrete label only, variation 2 means you use the score label only, and then, variation 3 means you use the score label but in the end you take the highest confidence to be your final predicted answer.

`-th` stands for `THRESHOLD`, in this flag, you can experimenting about numbers that pass the criteria for variation 2, score label.

## Location of predictions

The predictions will be stored in `python\results\{NAME}-{TIME_NOW}`. And then this code automatically push Trainer to `{USER_that_passed_by_TOKEN}/fine-tuned-{NAME}`.
