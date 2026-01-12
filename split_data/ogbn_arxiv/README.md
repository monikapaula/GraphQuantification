# OGBN_Arxiv 


Quantification task: estimate the prevalence of the 40 subject areas of arXiv CS papers, e.g., cs.AI, cs.LG, and cs.OS

## split_0
The model is trained on historical research papers from 1971 to 2017, validated on the year 2018, and tested on the most recent publications from 2019 to 2020.

| Dataset Split | Total Samples | Years Covered | Coverage (%) |
|:--------------| :--- | :--- |:-------------|
| **TRAIN**     | 90,941 | ≤ 2017 | 53.70%       |
| **VAL**       | 29,799 | 2018 | 17.60%       |
| **TEST**      | 48,603 | ≥ 2019 | 28.70%       |


## split_1 
The model is trained on the most recent publications from 2018 to 2020 and validated on papers from 2016 to 2017, before being tested on historical data spanning 1971 to 2015.

| Dataset Split | Total Samples | Years Covered | Coverage (%) |
|:--------------| :--- | :--- |:-------------|
| **TRAIN**     | 78,402 | $\geq$ 2018 | 46.30%       |
| **VAL**       | 37,781 | 2016–2017 | 22.31%       |
| **TEST**      | 53,160 | $\leq$ 2015 | 31.39%       |

## split_2
The model is trained on research papers spanning 1971 to 2016, validated on the 2017–2018 period, and evaluated on more recent publications through 2020. This temporal split is designed to test how well the model generalizes to the shifting trends of newer academic data.

| Dataset Split | Total Samples | Years Covered | Coverage (%) |
|:--------------| :--- | :--- |:-------------|
| **TRAIN**     | 69,499 | $\leq$ 2016 | 41.04%       |
| **VAL**       | 51,241 | 2017–2018 | 30.26%       |
| **TEST**      | 48,603 | $\geq$ 2019 | 28.70%       |




Data source: OGBN_Arxiv. Provided by the OGB (Open Graph Benchmark). Originally introduced in Open Graph Benchmark: Datasets for Machine Learning on Graphs (Hu et al., 2021). Available at: (https://ogb.stanford.edu/docs/nodeprop/)