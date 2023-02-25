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
python .\main_training_indonli.py -h
```

To run an experiments, you just only do this, you optionally need to passing arguments to {--learn-rate, --seed, --token} if you don't want using the default value provided:
```
python .\main_training_indonli.py -m indolem -d basic -e 16 -sa max
python .\main_training_indonli.py -m indolem -d translated -e 16 -sa max
python .\main_training_indonli.py -m indolem -d augmented -e 16 -sa max

python .\main_training_indonli.py -m indonlu -d basic -e 16 -sa max
python .\main_training_indonli.py -m indonlu -d translated -e 16 -sa max
python .\main_training_indonli.py -m indonlu -d augmented -e 16 -sa max

python .\main_training_indonli.py -m xlmr -d basic -e 16 -sa max
python .\main_training_indonli.py -m xlmr -d translated -e 16 -sa max
python .\main_training_indonli.py -m xlmr -d augmented -e 16 -sa max
```

The predictions will be stored in `python\results\{NAME}-{TIME_NOW}`. And, then this code automatically push Trainer to `{USER_that_passed_by_TOKEN}/fine-tuned-{NAME}`.

Thanks, 

Ravi.
