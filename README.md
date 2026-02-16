# A Collection of Benchmarks for Graph Quantification Learning under Real-world Distribution Shifts

This repository provides a comprehensive pipeline for training Graph Neural Networks (GNNs) for node classification and quantifying class prevalences using the QuaPy library. Created for a Bachelor thesis studying graph quantification under distribution shifts.

## Table of Contents
<!-- TOC -->
1. [Installation](#installation)
2. [Datasets and Splits](#datasets-and-splits)
3. [How to Run](#how-to-run)
    * [Phase 1: Training Classifiers](#training-classifiers)
    * [Phase 2: Quantification](#quantification)
4. [Results](#results)
<!-- TOC -->


## Installation

Ensure you have python 3.10+ and the necessary libraries installed:
```
pip install -r requirements.txt
```
This project uses Git LFS to manage large Pytorch ```.pt``` files and dataset splits. 

[IMPORTANT] You must have Git LFS installed on your system before pulling. If you don't have it, follow the [official installation guide here](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).

Once installed, run the following commands to download the data:
```
git lfs install
git lfs pull
```


## Datasets and Splits
The project presents four datasets each with a variety of feature-based distribution shifts. To learn more about the splits and class distributions click on the dataset:

| Dataset               | Nodes    | Edges     | Classes | Domain            |
| --------------------- |----------|-----------|---------|-------------------|
| [Presidential-Election](split_data/presidential_election/README.md) | 3K       | 18K       | 2       | Political Network |
| [Twitch-Gamers](split_data/twitch_gamers/README.md)         | 28K      | 92K       | 2       | Social Network    |
| [Deezer-Europe](split_data/deezer_europe/README.md)         | 1,9 - 9K | 31 - 153K | 2       | Social Network    |
| [OGBN-Arxiv](split_data/ogbn_arxiv/README.md)           | 169K     | 1.2M      | 40      | Citation Network  |
### To download the datasets, run:
```
python utils/dataset_manager.py 
```

**Note**: This step is not required to run the experiments. The experiment pipeline utilizes pre-computed masks included in the repository. You only need to run this script if you wish to access or inspect the original source datasets.

## How to Run
### Training Classifiers
The train_classifier.py script trains a selected model on a specific dataset and split. 

```
python train_classifier.py --datasets <dataset_name> --splits <split_1> <split_2> --models  MLP GCN SAGE --epochs 300
```

### Configuration options:
* --datasets: Name of the dataset (e.g., presidential_election, twitch_gamers, deezer_europe, ogbn_arxiv)
* --splits: One or more split names to iterate through
* --models: List of architectures to train (GCN, SAGE, MLP)
* --epochs: Number of training epochs

### Quantification 
Once models are trained, use quantify.py to estimate class prevalence in the test set. This phase supports standard aggregative quantifiers and custom graph-based SIS methods
```
python quantification/quantify.py --datasets <dataset_name> --models GCN --splits <split_name> --run_sis
```
### Configuration options:
* --datasets: Name of the datasets (e.g., presidential_election, twitch_gamers, deezer_europe, ogbn_arxiv)
* --models: List of architectures to train (GCN, SAGE, MLP)
* --splits: One or more split names to iterate through
* --run_sis: Flag to run SIS-ACC/PACC



### Supported Quantifiers:
* CC: Classify & Count
* ACC: Adjusted Classify & Count
* PCC: Probabilistic Classify & Count
* PACC: Probabilistic Adjusted Classify & Count
* KDEy: Kernel Density Estimation (using Quapy's GridSearch for bandwidth)
* SIS-ACC/PACC: Structural Importance Sampling 

## Results
CSV outputs track performance across all dataset/split/model combinations:
```
classification_results.csv:
Dataset,Split,Model,Macro_F1
presidential_election,temporal_shift,GCN,0.8234

quantification_results.csv:
Dataset,Split,Classifier,Method,MAE
presidential_election,temporal_shift,GCN,PACC,0.0342

```

## Project Structure
```text
├── train.py                # Main script for training the node classifiers
├── create_splits/          # Scripts to define dataset-specific splits
│   ├── presidential_election/
│   ├── twitch_gamers/
│   └── ...
├── models/                 # Model architectures (GCN, GraphSAGE, MLP)
├── utils/                  # Helper functions (Metrics, DataLoaders, SIS)
├── split_data/             # Processed datasets and split-specific READMEs
└── quantification/         # Quantification logic and evaluation
    ├── quantify.py         # Main quantification pipeline script
    └── results/            # Directory for experiment outputs 
```