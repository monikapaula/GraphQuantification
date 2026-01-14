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
| **TRAIN** | 1,256         | 138 (10.98%)       | 1,118 (89.01%)       |
| **VAL** | 314           | 31 (9.87%)         | 283 (90.13%)         |
| **TEST** | 1,571         | 384 (24.44%)       | 1,187 (75.56%)       |


## split_1: geographical 
Train on non_Eastern states, test on Eastern states

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- |:--------------|:-------------------|:---------------------|
| **TRAIN** | 1,741         | 245 (14.07%)       | **1,496 (85.93%)**   |
| **VAL** | 193           | 22 (11.39%)        | 171 (88.60%)         |
| **TEST** | 1,207         | 286 (23.69%)       | **921 (76.30%)**     |


## split_2: coastal
Train on interior states (mostly Republican) and test on coastal states (more Democrats)

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- |:--------------|:-------------------|:---------------------|
| **TRAIN** | 1,961         | 184 (9.38%)        | **1,777 (90.62%)**   |
| **VAL** | 291           | 62 (21.31%)        | 229 (78.69%)         |
| **TEST** | 889           | 307 (34.53%)       | **582 (65.46%)**      |

## split_3: metropolitan 
Create a split based on population/ density to simulate metropolitan vs non-metropolitan areas
Train on metropolitan counties (more Democrats) and test on non-metropolitan counties (more Republicans)

| Dataset Split | Total Samples | Class 0 (Democrat) | Class 1 (Republican) |
| :--- |:--------------|:-------------------|:---------------------|
| **TRAIN** | 989           | **338 (34.18%)**   | 651 (65.82%)         |
| **VAL** | 314           | 38 (12.10%)        | 276 (87.89%)         |
| **TEST** | 1,838         | **157 (8.54%)**    | 1,681 (91.46%)       |
