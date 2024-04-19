# Implementation of Hierarchical Attention Networks for Document Classification
Zichao Yang1 , Diyi Yang1 , Chris Dyer1 , Xiaodong He2 , Alex Smola1 , Eduard Hovy1

## Prerequisites
### Install Pretrained Word Vector GloVe 
```
cd Data
wget http://nlp.stanford.edu/data/glove.6B.zip
apt install unzip
unzip "glove.6B.zip"
```
Or download from https://nlp.stanford.edu/projects/glove/

## Dataset
**Yelp 2013 Dataset**
Can be found in https://zenodo.org/records/7555898
- The data used to train this model is 10% of *Yelp 2013*
- Preproccessed data is in folder Data
    - X is a list of *vectorized text*, a *vectorized text* is a list of *vectorized sentence*, and a sentence is list of 200-d *vectorized word*, using GloVe 6B-200d.
    - Y is a list of texts' labels. Label classes set: {0,1,2,3,4} 

## How to run
```
python3 main.py
```