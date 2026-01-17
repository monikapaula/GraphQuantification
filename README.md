# A Collection of Benchmarks for Graph Quantification Learning under Real-world Distribution Shifts

This repository provides a pipeline for training Graph Neural Networks (GNNs) for node classification and later quantifying the class prevalences using the QuaPy library. It was created for a Bachelor thesis 

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

Ensure you have python 3.8+ and the necessary libraries installed:
```
pip install torch torch-geometric pandas numpy quapy
```

## Datasets and Splits
The project presents four datasets each with a varity of feature-based distribution shifts
* [Presidential-Election](split_data/presidential_election/README.md)
* [Twitch-Gamers](split_data/twitch_gamers/README.md)
* [Deezer-Europe](split_data/deezer_europe/README.md)
* [OGBN-Arxiv](split_data/ogbn_arxiv/README.md)


### To download the datasets, run:
```
python utils/dataset_manager.py 
```

## How to Run
### Training Classifiers
The train.py script trains a selected model on a specific dataset and split. 

```
python train.py --dataset <dataset_name> --splits <split_1> <split_2> --models GCN SAGE --epochs 300
```

### Configuration options:
* --dataset: Name of the dataset (e.g., presidential_election, twitch_gamers, deezer_europe, ogbn_arxiv)
* --splits: One or more split names to iterate through
* --models: List of architectures to train (GCN, SAGE, MLP)
* --epochs: number of training epochs

### Quantification 
Once models are trained, use quantify.py to estimate class prevalence in the test set. This phase supports standard aggregative quantifiers and custom graph-based SIS methods
```
python quantify.py --datasets <dataset_name> --models GCN --splits <split_name> --run_sis
```
### Configuration options:
* --dataset: Name of the dataset (e.g., presidential_election, twitch_gamers, deezer_europe, ogbn_arxiv)
* -- models: List of architectures to train (GCN, SAGE, MLP)
* --splits: One or more split names to iterate through
* --run_sis: flag to run SIS-ACC/PACC



### Supported Quantifiers:
* CC: Classify & Count
* ACC: Adjusted Classify & Count
* PCC: Probabilistic Classify & Count
* PACC: Probabilistic Adjusted Classify & Count
* KDEy: Kernel Density Estimation (using Quapy's GridSearch for bandwidth)
* SIS-ACC/PACC: Structural Importance Sampling 