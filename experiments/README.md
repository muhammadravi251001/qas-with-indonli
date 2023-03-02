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

## Running experiments for training IndoNLI

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_training_indonli.py -h
```

To run this training IndoNLI experiments, you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size} if you don't want using the default value provided:
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

To run this fine-tuning QAS datasets experiments WITHOUT Intermediate Task Transfer Learning (ITTL), you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size} if you don't want using the default value provided:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqaid -e 16 -sa max -f no_ittl

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqaid -e 16 -sa max -f no_ittl

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max -f no_ittl
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqaid -e 16 -sa max -f no_ittl
```

It should be understood that when you use the `no_ittl` flag it means that you are fine-tuning the QAS dataset with the baseline flow, usually this is done as a reference whether the performance of a QAS is increasing or decreasing based on the "no_ittl" baseline flow.

And then, to run this fine-tuning QAS datasets experiments WITH Intermediate Task Transfer Learning (ITTL), you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size} if you don't want using the default value provided:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqaid -e 16 -sa max -f with_ittl

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqaid -e 16 -sa max -f with_ittl

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max -f with_ittl
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqaid -e 16 -sa max -f with_ittl
```

With use of `with_ittl` flag, that means you doing Intermediate Task Transfer Learning (ITTL). It's like you do fine-tuning twice, first with the SequenceClassification task with IndoNLI dataset and the second with the QuestionAnswering task with the QAS dataset that you have chosen yourself.

## Location of predictions

The predictions will be stored in `python\results\{NAME}-{TIME_NOW}`. And then this code automatically push Trainer to `{USER_that_passed_by_TOKEN}/fine-tuned-{NAME}`.
