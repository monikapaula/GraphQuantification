# Deezer_europe splits 

class balance between genders relatively equally distributed 
0: female = 15743
1: male = 12538

## split_0: 
The model is trained primarily on users who listen to female-dominated artists (Class 1).
The model is evaluated primarily on users who listen to male-dominated artists (Class 0).

### Class Balance Statistics

| Dataset Split | Total Samples | Class 0 (Count / %) | Class 1 (Count / %) |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 1,766 | 234 (13.25%) | **1,532 (86.75%)** |
| **VAL** | 196 | 26 (13.27%) | **170 (86.73%)** |
| **TEST** | 7,661 | **5,861 (76.50%)** | 1,800 (23.50%) |


## split_1:
The model is trained primarily on users who listen to male-dominated artists (Class 0).
The model is evaluated primarily on users who listen to female-dominated artists (Class 1).

### Class Balance Statistics

| Dataset Split | Total Samples | Class 0 (Count / %) | Class 1 (Count / %) |
| :--- | :--- | :--- | :--- |
| **TRAIN** | 6,895 | 5,275 (76.50%) | 1,620 (23.50%) |
| **VAL** | 766 | 586 (76.50%) | 180 (23.50%) |
| **TEST** | 1,962 | 260 (13.25%) | **1,702 (86.75%)** |
