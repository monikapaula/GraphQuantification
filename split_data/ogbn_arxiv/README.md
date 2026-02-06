# OGBN Arxiv 


Task: estimate the prevalence of the 40 subject areas of arXiv computer science papers

## split_0: Historical-to-Modern
The model is trained on historical research papers from 1971 to 2017, validated on the year 2018, and tested on the most recent publications from 2019 to 2020.

| Dataset Split | Total Samples | Years  | Coverage (%) |
|:--------------| :--- |:-------|:-------------|
| **TRAIN**     | 90,941 | ≤ 2017 | 53.70        |
| **VAL**       | 29,799 | 2018   | 17.60        |
| **TEST**      | 48,603 | ≥ 2019 | 28.70        |


## split_1: Modern-to-Historical
The model is trained on the most recent publications from 2018 to 2020 and validated on papers from 2016 to 2017, before being tested on historical data spanning 1971 to 2015.

| Dataset Split | Total Samples | Years       | Coverage (%) |
|:--------------| :--- |:------------|:-------------|
| **TRAIN**     | 78,402 | $\geq$ 2018 | 46.30        |
| **VAL**       | 37,781 | 2016–2017   | 22.31        |
| **TEST**      | 53,160 | $\leq$ 2015 | 31.39        |

## split_2: Mid-to-Both
The model is trained on research papers spanning 2014 to 2018, validated on the 2012 – 2013 period, and evaluated on  both recent and older publications.

| Dataset Split | Total Samples | Years                     | Coverage (%) |
|:--------------|:--------------|:--------------------------|:-------------|
| **TRAIN**     | 88,769        | 2014 - 2018               | 52.42        |
| **VAL**       | 14,570        | 2012 - 2013               | 8.60        |
| **TEST**      | 66,004        | $\leq$ 2011 & $\geq$ 2019 | 38.98        |




Data source: OGBN_Arxiv. Provided by the OGB (Open Graph Benchmark). Originally introduced in Open Graph Benchmark: Datasets for Machine Learning on Graphs (Hu et al., 2021). Available at: (https://ogb.stanford.edu/docs/nodeprop/)