import re
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split


def clean_text(text):
    punc = r"[^\w\s\.]"
    text = text.lower()
    text = re.sub('<sssss>','',text)
    text = re.sub('\.\.\.',' ',text)
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    text = re.sub(punc,'',text)
    return text.strip()

emmbed_dict = {}
with open('glove.6B.200d.txt','r') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:],'float32')
    emmbed_dict[word]=vector

text_file = open('Data/texts.txt','rt')
score_file = open('Data/score.txt','rt')

texts = []
scores = []
for line in text_file:
    texts.append(line)
for line in score_file:
    scores.append(int(line)-1)
    
X = []
max_text_len = 0
max_sentence_len = 0
for text in texts:
    text = clean_text(text)
    text = text.split('.')
    vectorized_text = []
    for sentence in text:
        vectorized_sentence = []
        sentence = sentence.strip()
        if sentence=='':
            continue
        else:
            sentence = sentence.split()
            for word in sentence:
              if(emmbed_dict.get(word) is None):
                continue
              else:
                vectorized_sentence.append(emmbed_dict[word])
        if(len(vectorized_sentence)>0):
            vectorized_text.append(vectorized_sentence)
        max_sentence_len = max(max_sentence_len,len(vectorized_sentence))

    X.append(vectorized_text)
    max_text_len = max(max_text_len,len(vectorized_text))

Y = torch.nn.functional.one_hot(torch.FloatTensor(scores).long())

# Take 10% of dataset 
X, _, Y, _ = train_test_split(X, Y, train_size=0.1)
with open('Data/X.pickle','wb') as file:
   pickle.dump(X,file)
with open('Data/Y.pickle','wb') as file:
   pickle.dump(Y,file)