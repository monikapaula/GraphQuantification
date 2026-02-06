# Presidential Election

Task: Predicting county-level voting outcomes (Democrat vs. Republican) in US elections.

Node features were sourced from the [US Election Dataset](https://www.kaggle.com/datasets/essarabi/ultimate-us-election-dataset?resource=download%5D) and edges from the [US Counties Covid-19 Dataset](https://www.kaggle.com/datasets/ady123/us-counties-covid19-dataset) on Kaggle.

The dataset is highly imbalanced:
- Number of Democrat counties: 553
- Number of Republican counties: 2588

## split_0: Rural-to-Urban
Simulates a typical election-night, in which smaller, more rural (low-density) counties tend to report their results earlier than lager counties. The model is trained on low-density counties and tested on more metropolitan (higher-density) areas.

| Dataset Split | Total Samples | Democrat (Count / %)    | Republican (Count / %) |
| :--- |:--------------|:-------------|:-----------------------|
| **TRAIN** | 1,256         | 138 (10.98%) | 1,118 (89.02%)         |
| **VAL** | 314           | 31 (9.87%)   | 283 (90.13%)           |
| **TEST** | 1,571         | 384 (24.44%) | 1,187 (75.56%)         |


## split_1: South-North 
The model is trained on southern states and exclusively evaluated on northern states.

| Dataset Split | Total Samples | Democrat (Count / %)    | Republican (Count / %)        |
| :--- |:--------------|:-------------|:-------------------|
| **TRAIN** | 1,620         | 278 (17.16%) | **1,342 (82.84%)** |
| **VAL** | 180           | 22 (12.22%)  | 158 (84.78%)       |
| **TEST** | 1,341         | 253 (18.87%) | **1088 (81.13%)**  |


## split_2: Inland-to-Coast
The split  creates a coastal versus interior partition, where the model gets trained on interior states (mostly Republican) and tested on coastal states.

| Dataset Split | Total Samples | Democrat (Count / %)    | Republican (Count / %)        |
| :--- |:--------------|:-------------|:-------------------|
| **TRAIN** | 1,341         | 100 (7.46%)  | **1,241 (92.54%)** |
| **VAL** | 291           | 62 (21.31%)  | 229 (78.69%)       |
| **TEST** | 1,509         | 391 (25.91%) | **1118 (74.09%)**  |

## split_3: Metropolitan-to-Rural 
This split simulates a common bias where only major urban centers are surveyed for their political preferences. Counties with metropolitan areas are assigned to the training set, while the remaining smaller counties form the validation and test set.

| Dataset Split | Total Samples | Democrat (Count / %)    | Republican (Count / %)    |
| :--- |:--------------|:------------------------|:---------------|
| **TRAIN** | 989           | **338 (34.18%)**        | 651 (65.82%)   |
| **VAL** | 314           | 38 (12.10%)             | 276 (87.90%)   |
| **TEST** | 1,838         | **157 (8.54%)**         | 1,681 (91.46%) |
