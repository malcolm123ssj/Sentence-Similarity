import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re
import math
from numpy import dot
from numpy.linalg import norm
import nltk
from nltk.stem import WordNetLemmatizer as Lemma
import collections
from collections import Counter

dataset = pd.read_csv('dataset.txt', delimiter = '\t', header = None, names = ['S.No', 'Sent1', 'Sent2', 'Rating', 'Output'])
dataset = dataset.drop(['Rating'], axis = 1)
dataset['Output'] = dataset['Output'].apply(lambda x:'CONTRADICTION' if x == 'CONTRADICTION' else 'NOT CONTRADICTION')

corpus1 = []
for x in dataset['Sent1']:
    sentence1 = x.lower()
    sentence1 = sentence1.split()
    wnl = Lemma()
    sentence1 = [wnl.lemmatize(word) for word in sentence1 if word not in stopwords.words('english')]
    sentence1 = ' '.join(sentence1)
    corpus1.append(sentence1)
    
corpus2 = []
for y in dataset['Sent2']:
    sentence2 = y.lower()
    sentence2 = sentence2.split()
    wnl = Lemma()
    sentence2 = [wnl.lemmatize(word) for word in sentence2 if word not in stopwords.words('english')]
    sentence2 = ' '.join(sentence2)
    corpus2.append(sentence2)

edit = []
for i in range(0,4500):
    ed = nltk.edit_distance(corpus1[i],corpus2[i])
    edit.append(ed)

jaccard = []
for i in range(0,4500):
    jd = nltk.jaccard_distance(set(corpus1[i]),set(corpus2[i]))
    jaccard.append(jd)
    
cosine_similar = []
WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator
    
def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)
 
for i in range(0,4500):
    vector1 = text_to_vector(corpus1[i])
    vector2 = text_to_vector(corpus2[i])
    cosine = get_cosine(vector1, vector2)
    cosine_similar.append(cosine)

LCS = []
def lcs(s1, s2):
    tokens1, tokens2 = s1.split(), s2.split()
    cache = collections.defaultdict(dict)
    for i in range(-1, len(tokens1)):
        for j in range(-1, len(tokens2)):
            if i == -1 or j == -1:
                cache[i][j] = 0
            else:
                if tokens1[i] == tokens2[j]:
                    cache[i][j] = cache[i-1][j-1] + 1
                else:
                    cache[i][j] = max(cache[i-1][j], cache[i][j-1])
    return cache[len(tokens1)-1][len(tokens2)-1]

for i in range(0,4500):
    s = lcs(corpus1[i],corpus2[i])
    similar = s/max(len(corpus1[i]),len(corpus2[i]))
    LCS.append(similar)
