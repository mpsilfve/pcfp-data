# pcfp-data

This repository contains datasets for experimentation on the paradigm cell-filling task (PCFP) where the task is to fill 
in the missing forms in a set of morphological inflection tables. For example,

```
try    INF                       try     INF
       PRES+3+SG         =>      tries   PRES+3+SG
       PRES+PCPLE                trying  PRES+PCPLE
tried  PAST                      tried   PAST
```

These datasets were used in the experiments for the paper 

Miikka Silfverberg and Mans Hulden. *An Encoder-Decoder Approach to the Paradigm Cell Filling Problem*. EMNLP 2018.

### Training and test data

The directory `data` contains inflection tables for eight languages: Finnish, French, Georgian, German, Latin,
Latvian, Spanish and Turkish. These were created based on the [Unimorph](http://unimorph.org/) data sets for each language.
Originally, the inflection tables have been crawled from Wiktionary.

Each file `LAN.um.POS.txt` contains 1,000 inflection tables randomly sampled from the Unimorph data set for language `LAN` and 
part-of-speech `POS` (`LAN` is one of `de`, `fi`, `fr`, `kat`, `la`, `lv`, `es` or `tr`, and `POS` is either `N` for nouns or
`V` for verbs). The files `LAN.um.POS.n` contain the same inflection tables but these tables only show `n`=1,2 or 3 randomly 
sampled forms. Your task is to use each file `LAN.um.POS.n.txt` to learn a model which can fill in the missing forms in that file.

Each file `LAN.um.POS.top.txt` contains a varying number of inflection tables. These have been chosen to include Unimorph
tables which include at least one word form from the list of 10,000 most frequent word forms for language `LAN` based on
word frequencies in Wikipedia for lan `LAN`. The file `LAN.um.POS.top.10000.txt` contains the same inflection tables but only
the most frequent word forms are shown. The task is to fill in the missing forms.

### System outputs

The directory `outputs` contains the outputs of the models presented in the EMNLP paper. Each file `LAN.um.POS.n.vote.res.txt`
contains outputs for input file `LAN.um.POS.n.txt`. The files `LAN.um.POS.top.res.txt` correspond to the test file 
`LAN.um.POS.top.10000.txt`. The files `LAN.um.POS.n.baseline.res.txt` and `LAN.um.POS.top.baseline.res.txt` are analogous files for the 
baseline system.

### Code

The `src` directory contains code and documentation for running the experiments in Silfverberg and Hulden (2018).
