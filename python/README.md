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
cd python
pip install -r requirements.txt
```

## Running experiments for training IndoNLI

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_training_indonli.py -h
```

To run this training IndoNLI experiments, you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size} if you don't want using the default value provided:
```
python main_training_indonli.py -m indolem -d basic -e 16 -sa max
python main_training_indonli.py -m indolem -d translated -e 16 -sa max
python main_training_indonli.py -m indolem -d augmented -e 16 -sa max

python main_training_indonli.py -m indonlu -d basic -e 16 -sa max
python main_training_indonli.py -m indonlu -d translated -e 16 -sa max
python main_training_indonli.py -m indonlu -d augmented -e 16 -sa max

python main_training_indonli.py -m xlmr -d basic -e 16 -sa max
python main_training_indonli.py -m xlmr -d translated -e 16 -sa max
python main_training_indonli.py -m xlmr -d augmented -e 16 -sa max
```

## Running experiments for fine-tuning dataset QAS

Please, check the arguments that can be passed to this code; the datatype, arguments choice, and default value.
```
python main_fine_tuning_qas_dataset.py -h
```

To run this fine-tuning dataset QAS experiments, you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token, --batch_size} if you don't want using the default value provided:
```
python main_fine_tuning_qas_dataset.py -m indolem -d squadid -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m indolem -d idkmrc -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m indolem -d tydiqa -e 16 -sa max

python main_fine_tuning_qas_dataset.py -m indonlu -d squadid -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m indonlu -d idkmrc -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m indonlu -d tydiqa -e 16 -sa max

python main_fine_tuning_qas_dataset.py -m xlmr -d squadid -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m xlmr -d idkmrc -e 16 -sa max
python main_fine_tuning_qas_dataset.py -m xlmr -d tydiqa -e 16 -sa max
```

The predictions will be stored in `python\results\{NAME}-{TIME_NOW}`. And, then this code automatically push Trainer to `{USER_that_passed_by_TOKEN}/fine-tuned-{NAME}`.

Thanks, 

Ravi.
