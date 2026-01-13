# Presidential_election

Predicting voting outcomes (Democrats vs. Republicans) for US counties 

features from [https://www.kaggle.com/datasets/essarabi/ultimate-us-election-dataset?resource=download%5D] 

edges from [https://www.kaggle.com/datasets/ady123/us-counties-covid19-dataset]

The dataset is highly imbalanced:
- Number of Democrat counties: 553
- Number of Republican counties: 2588

## split_0: random
Simulates a standard election with scattered missing results

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- |:--------------|:-------------------|:---------------------|
| **TRAIN** | 1,256         | 126 (10.03%)       | 1,130 (89.97%)       |
| **VAL** | 314           | 41 (13.06%)        | 273 (86.94%)         |
| **TEST** | 1,571         | 386 (24.57%)       | 1,185 (75.43%)       |


## split_1: geographical 
Train on non_Eastern states, test on Eastern states

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 2,105 | 242 (11.50%) | **1,863 (88.50%)**|
| **VAL** | 233 | 72 (30.90%) | 161 (69.10%) |
| **TEST** | 803 | 239 (29.76%) | **564 (70.24%)** |


## split_2: coastal
Train on interior states (mostly Republican) and test on coastal states (more Democrats)

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 2,844 | 485 (17.05%) | **2,359 (82.95%)** |
| **VAL** | 297 | 68 (22.90%) | 229 (77.10%) |
| **TEST** | 342 | 274 (80.12%) | **68 (19.88%)** |

## split_3: metropolitan 
Create a split based on population/ density to simulate metropolitan vs non-metropolitan areas
Train on metropolitan counties (more Democrats) and test on non-metropolitan counties (more Republicans)

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 2,094 | **461 (22.02%)** | 1,633 (77.98%) |
| **VAL** | 314 | 21 (6.69%) | 293 (93.31%) |
| **TEST** | 628 | **60 (9.55%)** | 568 (90.45%) |
