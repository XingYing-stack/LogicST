# LogicST: A Logical Self-Training Framework for Document-Level Relation Extraction with Incomplete Annotations

This repository contains the code for the paper "LogicST: A Logical Self-Training Framework for Document-Level Relation Extraction with Incomplete Annotations," which is currently under review.

## Requirements

To run this code, you will need the following Python packages:

- `apex==0.1`
- `bibtexparser==1.4.1`
- `dill==0.3.4`
- `matplotlib==2.2.3`
- `numpy==1.19.5`
- `opt_einsum==3.3.0`
- `pandas==1.1.5`
- `pyecharts==2.0.3`
- `scipy==1.5.4`
- `torch==1.7.1+cu101`
- `tqdm==4.62.1`
- `transformers==4.18.0`
- `ujson==4.0.2`

## Dataset

The datasets used in this project can be downloaded from the following links:

- The [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded following the instructions [here](https://github.com/thunlp/DocRED/tree/master/data).
- The [Re-DocRED](https://aclanthology.org/2022.emnlp-main.580.pdf) dataset can be downloaded following the instructions [here](https://github.com/tonytan48/Re-DocRED).
- The [DocRED_ext](https://arxiv.org/pdf/2210.08709) dataset can be downloaded following the instructions [here](https://github.com/www-Ye/SSR-PU).
- The [DocGNRE](https://aclanthology.org/2023.emnlp-main.334.pdf) dataset can be downloaded following the instructions [here](https://github.com/bigai-nlco/DocGNRE).
- The original [DWIE](https://arxiv.org/pdf/2009.12626) dataset can be downloaded following the instructions [here](https://github.com/klimzaporojets/DWIE). The pre-processing process is the same as [LogiRE](https://aclanthology.org/2021.emnlp-main.95.pdf), more details can be found [here](https://github.com/rudongyu/LogiRE). We also provide a script `./dataset/dwie/build_incomplete_dataset.py` to generate the incompletely labeled dataset.


```
LogicST
 |-- dataset
 |    |-- docred
 |    |    |-- rel_info.json        
 |    |    |-- rel2id.json        
 |    |    |-- train_annotated.json (DocRED)     
 |    |    |-- train_ext.json (DocRED_ext)
 |    |    |-- dev_revised.json (Re-DocRED)
 |    |    |-- test_revised.json (Re-DocRED)
 |    |    |-- re_docred_test_data_enhancement_human.json (DocGNRE)
 |    |-- dwie
 |    |    |-- train_annotated.json 
 |    |    |-- train_incomplete_0.2.json
 |    |    |-- train_incomplete_0.4.json
 |    |    |-- train_incomplete_0.6.json
 |    |    |-- train_incomplete_0.8.json
 |    |    |-- meta
 |    |    |    |--ner2id.json
 |    |    |    |--rel2id.json
 |    |    |    |--word2id.json
 |    |    |    |--vec.npy
```

## Logical Rules
We use the rule miner from [MILR](https://aclanthology.org/2022.emnlp-main.704.pdf). More details can be found in [link](https://github.com/XingYing-stack/MILR) and the python file `./mine_rule.py`.


## Training
### DocRED
Train the BERT / RoBERTa model on DocRED with the following command:

```bash
>> sh scripts/run_bert_docred.sh $i  # for BERT trained on cuda:i
>> sh scripts/run_roberta_docred.sh $i  # for RoBERTa trained on cuda:i
```

### DWIE
```bash
>> scripts/run_bert_dwie.sh $i $j  # for BERT trained on positive sampling ratio with $j on cuda:i
```

## Evaluating Models
The save_path is recorded in the training log file. 

You can load the model by the `--load_path` argument, then the code will skip training and evaluate the saved model on benchmarks.

To facilitate reproduction, we provide two checkpoints trained on [DocRED](https://drive.google.com/file/d/1h4iIf2k9OsIU2RqF0HsVpc8DV1PLwxGW/view?usp=drive_link) and [DWIE with 40% positive sampling](https://drive.google.com/file/d/1sSe6ASVQ5HqL6mDGxPFodIX8bBrilPyE/view?usp=drive_link) with BERT-base-uncased.


## Predictions
We provide various frameworks' predictions on DocRED's test set in `./results`. These include vanilla ATLOP, negative sampling, CAST, P3M, LogicST.


## Logs
We provide logs trained on the training set of DocRED and DWIE with 40% sampling ratios in `./logs`.

