=== Introduction

This is deep learning code for running experiments he Paradigm Cell Filling Problem (PCFP). The code is meant for running the experiments in the paper

Miikka Silfverberg and Mans Hulden. *An Encoder-Decoder Approach to the Paradigm Cell Filling Problem*. EMNLP 2018.

In the PCFP, you are given a number of partial inflection tables. For example,

```
walk         V;INF
             PRES;3SG
walk         PRES;NON3SG
             PAST
             PCPLE;PAST
walking      PCPLE;PRES```

and the objective is to fill in the missing forms in all the paradigms

```
walk         V;INF
walks        PRES;3SG
walk         PRES;NON3SG
walked       PAST
walked       PCPLE;PAST
walking      PCPLE;PRES```

We experiment with three settings.

You should use 
   * `train_n_gt_1.py` and `test_n_gt_1.py`, respectively, for training and testing models when 

=== Training Models

the `train_XYZ.py` scripts are used for training models. 
