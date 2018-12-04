# AutoEncoder-for-recommend-system
this is tensorflow implementation for paper :"AutoRec: Autoencoders Meet Collaborative Filtering"(https://dl.acm.org/citation.cfm?id=2740908.2742726)

AE for recommend system is similar to collaborative filtering , so icf-AE is better than ucf-AE and this code is icf version

dataset is movielens-1M

first we need split raw data into train set and test set, the script can be found in ./data/ml-1m/data.py, there is no script for ml-100k
so if you want to run in ml-100k you must write a special script for ml-100k acording ./data/ml-1m/data.py

## split raw data into train set and test set
1. get matrix from raw data
2. select a certain percentage of data from each user for test data, and the rest is train data

## run code
```
main.py
```
