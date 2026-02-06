# Deezer_europe splits 

The Deezer Europe dataset consists of 28,281 users with a base gender distribution of:
- Class 0 (Female): 15743
- Class 1 (male): 12538

The task is to quantify the gender across splits.

## 1. Gender-Biased Splits (Artist Preference): 

### split_0: Female-to-Male
The model is trained primarily on users who listen to female-dominated artists, while the model is evaluated primarily on users who listen to male-dominated artists. The split does not include featureless nodes (users without artist history) into the training set.

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- |:--------------|:-------------------|:-----------------|
| **TRAIN** | 10,496        | 7,535 (71.79%)     | 2,961 (28.21%)   |
| **VAL** | 2,624         | 1,914 (72.94%)     | 710 (27.06%)     |
| **TEST** | 8,576         | 2,597 (30.28%)     | 5,979 (69.72%)   |


### split_1: Male-to-Female
The model is trained primarily on users who listen to male-dominated artists and evaluated on users who listen to female-dominated artists. The split does not include featureless nodes (users without artist history) into the training set.

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- |:--------------|:-------------------|:-----------------|
| **TRAIN** | 8,030         | 2,704 (33.67%)     | 5,326 (66.33%)   |
| **VAL** | 2,007         | 676 (33.68%)       | 1,331 (66.32%)   |
| **TEST** | 12,062        | 8,888 (73.69%)     | 3,174 (26.31%)   |


## 2. Behavioral & Popularity Splits

### split_2: Mainstream-to-Niche
I split the users based on the overall popularity of the artists they listen to. This separates the data into distinct groups, such as 'mainstream' listeners versus those who prefer niche or less famous music. The split does not include featureless nodes (users without artist history) into the training set.

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- | :--- |:-------------------|:-----------------|
| **TRAIN** | 13,674 | 7,220 (52.80%)     | 6,454 (47.20%)   |
| **VAL** | 3,418 | 1,744 (51.02%)     | 1,674 (48.98%)   |
| **TEST** | 5,030 | 3,319 (65.98%)     | 1,711 (34.02%)   |

## 3. Includes Featureless Nodes

### split_3: Female-to-Male
Same as split_0 but includes featureless nodes. 

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- |:--------------|:-------------------|:-----------------|
| **TRAIN** | 15,764        | 10,519 (66.73%)    | 5,245 (33.27%)   |
| **VAL** | 3,941         | 2,627 (66.66%)     | 1,314 (33.34%)   |
| **TEST** | 8,576        | 2,597 (30.28%)     | 5,979 (69.72%)   |


### split_4: Male-to-Female
Same as split_1 but includes featureless nodes.

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- |:--------------|:-------------------|:-----------------|
| **TRAIN** | 12,976        | 5,489 (42.30%)     | 7,487 (57.70%)   |
| **VAL** | 3,243         | 1,366 (42.12%)     | 1,877 (57.88%)   |
| **TEST** | 12,062        | 8888 (73.69%)      | 3,174 (26.31%)   |

### split_5: Mainstream-to-Niche
Same as spit_3 but includes featureless nodes. 

| Dataset Split | Total Samples | Female (Count / %) | Male (Count / %) |
| :--- | :--- |:-------------------|:-----------------|
| **TRAIN** | 18,601 | 9,959 (53.54%)     | 8,642 (46.46%)   |
| **VAL** | 4,650 | 2,465 (53.01%)     | 2,185 (46.99%)   |
| **TEST** | 5,030 | 3,319 (65.98%)     | 1,711 (34.02%)   |

Data Source: Deezer Europe Dataset. Provided by the Stanford Network Analysis Project (SNAP). Originally introduced in: Multi-scale Attributed Node Embedding (Rozemberczki et al., 2019). 
Available at: (https://snap.stanford.edu/data/feather-deezer-social.html)
