# Introduction (Waiting for completion, I am collecting the trained models and uploading them)
This repository contains the data, code, and models for our paper (SDJC). It is built upon Pytorch and Huggingface.
Due to ethical issues, the educational domain corpus, and downstream task QA bot data are not public now. Please pay your patience for the ethic approvement. (If you want to test your model on the QAbot data, feel free to contact me, we can test your model on our server. My email address: cyjk2101@mail4.doshisha.ac.jp)

## Overview
Several backbone models pre-trained on general domain datasets can encode a sentence into a widely useful embedding. Such sentence embeddings can be further enhanced by domain adaptation that adapts a backbone model to a specific domain. However, domain adaptation for low-resource languages like Japanese is often difficult due to the scarcity of large-scale labeled datasets. To overcome this, this paper introduces SDJC (Self-supervised Domain adaptation for Japanese sentence embeddings with Contrastive learning) that utilizes a data generator to generate sentences, which have the same syntactic structure to a sentence in an unlabeled specific domain corpus but convey different semantic meanings. Generated sentences are then used to boost contrastive learning that adapts a backbone model to accurately discriminate sentences in the specific domain.

![overall image](/SDJC_overview.png)

## Requirements
We recommend the following dependencies.
+ Python 3.8
+ Pytorch 1.9
+ transformers 4.22.2
+ datasets 2.3.2
+ spaCy 3.3.1
+ GiNZA 5.1.2

## Japanese STS benchmark
Another problem of Japanese sentence representation learning is the difficulty of evaluating existing embedding methods due to the lack of benchmark datasets. Thus, we establish a comprehensive Japanese Semantic Textual Similarity (STS) benchmark on which various embedding models are evaluated.

We use the SentEval toolkit to evaluate embedding models on the Japanese STS benchmark which has been combined in our published [SentEval toolkit](/SentEval/data/downstream).

### Evaluation
You can evaluate any Japanese sentence embedding models following the commands below:
```
python evaluation.py \
    --model_name_or_path <your_model_dir> \
    --pooler  <cls|cls_before_pooler|avg|avg_top2|avg_first_last> \
    --task_set <sts|transfer|full> \
    --mode test
```

### Model List
The Japanese sentence embedding models trained by us are listed as follows.



You can also check other Japanese sentence embedding models trained by us in our previous work.
| Model |
| ----- |
| [MU-Kindai/Japanese-SimCSE-BERT-base-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-unsup) |
| [MU-Kindai/Japanese-SimCSE-BERT-large-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-unsup) |
| [MU-Kindai/Japanese-SimCSE-RoBERTa-base-unsup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-RoBERTa-base-unsup) |
| [MU-Kindai/Japanese-MixCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-MixCSE-BERT-base) |
| [MU-Kindai/Japanese-MixCSE-BERT-large](https://huggingface.co/MU-Kindai/Japanese-MixCSE-BERT-large) |
| [MU-Kindai/Japanese-DiffCSE-BERT-base](https://huggingface.co/MU-Kindai/Japanese-DiffCSE-BERT-base) |
| [MU-Kindai/SBERT-JSNLI-base](https://huggingface.co/MU-Kindai/SBERT-JSNLI-base) |
| [MU-Kindai/SBERT-JSNLI-large](https://huggingface.co/MU-Kindai/SBERT-JSNLI-large) |
| [MU-Kindai/Japanese-SimCSE-BERT-base-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-base-sup) |
| [MU-Kindai/Japanese-SimCSE-BERT-large-sup](https://huggingface.co/MU-Kindai/Japanese-SimCSE-BERT-large-sup) |

## SDJC
### Download data
Wikipedia data and JSNLI data for contrastive learning can be downloaded from [here](/data/download_wiki.sh) and [here](/data/download_nli.sh):
```
wget https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/wiki1m.txt
wget https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/nli_for_simcse.csv 
```
The target domain corpus used in our paper can be downloaded from [here](/data/clinic_corpus.txt) and [here](/data/QAbot_corpus.txt).
Due to ethical issues, the educational domain corpus is not public now, but you can check the examples in our paper.

### Data generator fine-tune and generate contradictory data
You can finetune the data generator using the code referring [this one](T5_denoising_training_clinic_domain.py). 

You can generate contradictory data referring the following code from [here](/data_generation_for_unsup.ipynb).

You can download and directly use the synthetic data in target domain for contrastive learning from the following list.
|Synthetic Data|
|--------------|
|[clinic_domain_top4](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/clinic_shuffle_for_simcse_top4.csv)|
|[clinic_domain_top5](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/clinic_shuffle_for_simcse_top5.csv)|
|[clinic_domain_top6](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/clinic_shuffle_for_simcse_top6.csv)|
|[education_domain_top4](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/qa_shuffle_for_simcse_top4.csv)|
|[education_domain_top5](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/qa_shuffle_for_simcse_top5.csv)|
|[education_domain_top6](https://huggingface.co/datasets/MU-Kindai/datasets-for-JCSE/blob/main/qa_shuffle_for_simcse_top6.csv)|

### Training
Run `train.py`. You can define different hyperparameters in your own way.
In our experiments, we save multiple checkpoints and find the best one among saved ones.
```
python train.py \
    --model_name_or_path <your_model_dir> \
    --train_file <data_dir> \
    --output_dir <model_output_dir>\
    --num_train_epochs <training_epoch> \
    --per_device_train_batch_size 512 \
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5 \
    --max_seq_length 32 \
    --save_strategy steps \
    --save_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --hard_negative_weight 1 \
    --temp 0.05 \
    --do_train \
```
Arguments used to train our models:
| Method | Arguments |
| ------ | --------- |
| [MU-Kindai/JCSE-clinic-stage1-base](https://huggingface.co/MU-Kindai/JCSE-clinic-stage1-base) | `--train_file clinic_shuffle_for_simcse_top4.csv --learning_rate 5e-5 --hard_negative_weight 0`|
| [MU-Kindai/JCSE-clinic-final-base](https://huggingface.co/MU-Kindai/JCSE-clinic-final-base) | `--train_file nli_for_simcse.csv --learning_rate 5e-5 --hard_negative_weight 1`|
| [MU-Kindai/JCSE-clinic-stage1-large](https://huggingface.co/MU-Kindai/JCSE-clinic-stage1-large) | `--train_file clinic_shuffle_for_simcse_top5.csv --learning_rate 1e-5 --hard_negative_weight 0`|
| [MU-Kindai/JCSE-clinic-final-large](https://huggingface.co/MU-Kindai/JCSE-clinic-final-large) | `--train_file nli_for_simcse.csv --learning_rate 1e-5 --hard_negative_weight 1`|
| [MU-Kindai/JCSE-edu-stage1-base](https://huggingface.co/MU-Kindai/JCSE-edu-stage1-base) | `--train_file qa_shuffle_for_simcse_top4.csv --learning_rate 5e-5 --hard_negative_weight 0`|
| [MU-Kindai/JCSE-edu-final-base](https://huggingface.co/MU-Kindai/JCSE-edu-final-base) | `--train_file nli_for_simcse.csv --learning_rate 5e-5 --hard_negative_weight 1`|
| [MU-Kindai/JCSE-edu-stage1-large](https://huggingface.co/MU-Kindai/JCSE-edu-stage1-large) | `--train_file qa_shuffle_for_simcse_top6.csv --learning_rate 1e-5 --hard_negative_weight 0`|
| [MU-Kindai/JCSE-edu-final-large](https://huggingface.co/MU-Kindai/JCSE-edu-final-large) | `--train_file nli_for_simcse.csv --learning_rate 1e-5 --hard_negative_weight 1`|

### Evaluation
For the clinic domain STS tasks in our paper, you can evaluate the embedding models following the commands below:
```
python evaluation.py \
    --model_name_or_path <your_model_dir> \
    --pooler  avg \
    --task_set transfer \
    --mode test
```

For the education domain information retrieval tasks in our paper, you can evaluate the embedding models following the commands below:
```
cd QAbot_task_eva
python main.py\
    --model_name_or_path <your_model_dir>
```

Warning: Due to ethical issues, the QAbot data is not public now, but you can check the examples in our paper. If you want to test your model on the QAbot data, feel free to contact me, and we can test your model on our server.

## Relevant Content Words
For the relevant content words experiments in our paper, you can check and refer the codes and examples from [here](/relevant_content_words).

## Contact
If you have any questions, feel free to contact me. 
Email address: cyjk2101@mail4.doshisha.ac.jp
Lab homepage: https://ccilab.doshisha.ac.jp/

# Citation
If this work is helpful, please cite the following paper:
```
@article{chen2025domain,
  title={Domain Adaptation for Japanese Sentence Embeddings with Contrastive Learning based on Synthetic Sentence Generation},
  author={Chen, Zihao and Handa, Hisashi and Ohsaki, Miho and Shirahama, Kimiaki},
  journal={arXiv preprint arXiv:2503.09094},
  year={2025}
}
```

This work is based on our previous work:
```
@article{chen2023jcse,
  title={JCSE: contrastive learning of japanese sentence embeddings and its applications},
  author={Chen, Zihao and Handa, Hisashi and Shirahama, Kimiaki},
  journal={arXiv preprint arXiv:2301.08193},
  year={2023}
}
```
