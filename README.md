# Indonesia's Question Answering System With Utilizing Of Indonesia Natural Language Inference (IndoNLI): An Experiment Of Natural Language Inference For Indonesian.

This is a repository code accompanying my final thesis and soon-to-be paper `TODO`. The experiments can be found under the `experiments` directory:

- `main_fine_tuning_qas_dataset.py`: this file contains experiment regarding to Sequence Classification task with use various kind of IndoNLI; like: Basic --the one that available in [Hugging Face](https://huggingface.co/datasets/indonli), Translated --the one that available `translate_train.tar.gz` in [this repo](https://github.com/ir-nlp-csui/indonli/tree/main/data), and lastly: Augmented --this is a concatenation from Basic & Translated version above.

- `main_training_indonli.py`: this file contains experiment regarding to Question Answering task with use two kind of experiments; without Intermediate Task Transfer Learning and with Intermediate Task Transfer Learning. You can read further by reading the `README.md` in `experiments` folder.

- And so on, :grin::grin::grin:

The experiment code can be found under `experiment` directory, please check the related [README](https://github.com/muhammadravi251001/qas-with-indonli/blob/main/experiments/README.md) file.

The utilites can be found under the `utilities` directory, `utilities` consist all of the support all of the `experiment` things. Such as: Exploratory Data Analysis of various of Question Answering datasets, code to augmented basic & translated IndoNLI, etc. Pplease check the related [README](https://github.com/muhammadravi251001/qas-with-indonli/blob/main/utilities/README.md) file.

All of my experiment --in my local UI campus server, there are also several runs on the MBZUAI campus server on the `@afaji` (one of my supervisor) account; not the `@muhammadravi251001` account-- is also available in Hugging Face (https://huggingface.co/muhammadravi251001).
