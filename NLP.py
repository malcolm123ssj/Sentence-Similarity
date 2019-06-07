import pandas as pd
from nltk.corpus import stopwords
import re
import math
import nltk
from nltk.stem import WordNetLemmatizer as Lemma
import collections
from collections import Counter
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.utils import resample
import contractions
from nltk.util import ngrams
from fuzzywuzzy import fuzz
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
import sklearn.metrics as SM

dataset = pd.read_csv('dataset.txt', delimiter = '\t', header = None, names = ['S.No', 'Sent1', 'Sent2', 'Rating', 'Output'])
dataset = dataset.drop(['Rating'], axis = 1)
dataset['Output'] = dataset['Output'].apply(lambda x:'CONTRADICTION' if x == 'CONTRADICTION' else 'NOT CONTRADICTION')
label = LE()
dataset['Output'] = label.fit_transform(dataset['Output'])

df_majority = dataset[dataset['Output'] == 1]
df_minority = dataset[dataset['Output'] == 0]
df_minority_upsampled = resample(df_minority, replace=True, n_samples = 3500)
df_upsampled = pd.concat([df_majority, df_minority_upsampled]) 

A = ['no','not','never','nothing','against','but','very','little','most','off','until','above','below','nor','none','all','same','different']

corpus1 = []
for x in df_upsampled['Sent1']:
    sentence1 = x.lower()
    sentence1 = re.sub('[^a-zA-Z0-60000]',' ',sentence1)
    sentence1 = contractions.fix(sentence1)
    sentence1 = sentence1.split()
    wnl = Lemma()
    sentence1 = [wnl.lemmatize(word) for word in sentence1 if word not in stopwords.words('english') or word in A]
    sentence1 = ' '.join(sentence1)
    corpus1.append(sentence1)
    
corpus2 = []
for y in df_upsampled['Sent2']:
    sentence2 = y.lower()
    sentence1 = re.sub('[^a-zA-Z0-60000]',' ',sentence2)
    sentence2 = contractions.fix(sentence2)
    sentence2 = sentence2.split()
    wnl = Lemma()
    sentence2 = [wnl.lemmatize(word) for word in sentence2 if word not in stopwords.words('english') or word in A]
    sentence2 = ' '.join(sentence2)
    corpus2.append(sentence2)

edit1 = []
for i in range(0,7335):
    ed = nltk.edit_distance(corpus1[i],corpus2[i])
    edit1.append(ed)

jaccard1 = []
for i in range(0,7335):
    jd = nltk.jaccard_distance(set(corpus1[i]),set(corpus2[i]))
    jaccard1.append(jd)
    
def euclidDistance(v1 ,v2):
    return math.sqrt(sum((v1[k] - v2[k])**2 for k in set(v1.keys()).intersection(set(v2.keys()))))

def speechCount(sent):
    tokens = nltk.word_tokenize(sent)
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    
    res ={'NN':0 ,'JJ' :0 , 'VB' :0}
    
    for word ,tag in tags:
        if tag in ['NN','NNS','NNP','NNPS']:
            res['NN']+=1;
        elif tag in ['JJ','JJR','JJS']:
            res['JJ']+=1
        elif tag in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            res['VB']+=1
    
    total = sum(res.values())
    return dict((word, float(count)/total) for word,count in res.items())

cosine_similar1 = []
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
 
for i in range(0,7335):
    vector1 = text_to_vector(corpus1[i])
    vector2 = text_to_vector(corpus2[i])
    cosine = get_cosine(vector1, vector2)
    cosine_similar1.append(cosine)

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

for i in range(0,7335):
    s = lcs(corpus1[i],corpus2[i])
    similar = s/max(len(corpus1[i]),len(corpus2[i]))
    LCS.append(similar)
        
def Intersection(lst1, lst2): 
    return set(lst1).intersection(lst2) 

def get_tuples_nosentences(txt, NGRAM):
    if not txt: 
        return None
    ng = ngrams(txt.split(), NGRAM)
    return list(ng)

def jaccard_distance_ngrams(a,b):
    a = set(a)
    b = set(b)
    return 1.0 * len(a&b)/len(a|b)

def cosine_similarity_ngrams(a,b):
    vec1 = Counter(a)
    vec2 = Counter(b)
    
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return float(numerator) / denominator

cosine_similar2 = []
for i in range(0,7335):
    a = get_tuples_nosentences(corpus1[i],2)
    b = get_tuples_nosentences(corpus2[i],2)
    cs = cosine_similarity_ngrams(a,b)
    cosine_similar2.append(cs)
  
jaccard2 = []
for i in range(0,7335):
    a = get_tuples_nosentences(corpus1[i],2)
    b = get_tuples_nosentences(corpus2[i],2)
    jd = jaccard_distance_ngrams(a,b)
    jaccard2.append(jd)

overlap1 = []
for i in range(0,7335):
    a = get_tuples_nosentences(corpus1[i],1)
    b = get_tuples_nosentences(corpus2[i],1)
    c = Intersection(a,b)
    ov = len(set(c))/min(len(set(a)),len(set(b)))
    overlap1.append(ov)

overlap2 = []
for i in range(0,7335):
    a = get_tuples_nosentences(corpus1[i],2)
    b = get_tuples_nosentences(corpus2[i],2)
    c = Intersection(a,b)
    ov = len(set(c))/min(len(set(a)),len(set(b)))
    overlap2.append(ov)

fuzzy = []
for i in range(0,7335):
    fov = fuzz.token_set_ratio(df_upsampled['Sent1'].iloc[i],df_upsampled['Sent2'].iloc[i])
    fuzzy.append(fov)
    
euclid = []
for i in range(0,7335):
    vector1 = text_to_vector(corpus1[i])
    vector2 = text_to_vector(corpus2[i])
    c = euclidDistance(vector1,vector2)
    euclid.append(c)
    
parts_of_speech = []
for i in range(0,7335):
    d1 = speechCount(df_upsampled['Sent1'].iloc[i])
    d2 = speechCount(df_upsampled['Sent2'].iloc[i])
    d3 = [abs(d1[key] - d2.get(key, 0)) for key in d1.keys()]
    parts_of_speech.append(d3)  
    
noun, adjective, verb = [], [], []    
for i in range(0,7335):
    n = parts_of_speech[i][0]
    noun.append(n)
    ad = parts_of_speech[i][1]
    adjective.append(ad)
    ve = parts_of_speech[i][2]
    verb.append(ve)
    
df_upsampled.insert(3, "Edit", edit1, True) 
df_upsampled.insert(3, "Jaccard1", jaccard1, True) 
df_upsampled.insert(3, "Cosine1", cosine_similar1, True) 
df_upsampled.insert(3, "LCS", LCS, True) 
df_upsampled.insert(3, "Jaccard2", jaccard2, True) 
df_upsampled.insert(3, "Cosine2", cosine_similar2, True) 
df_upsampled.insert(3, "Overlap1", overlap1, True) 
df_upsampled.insert(3, "Overlap2", overlap2, True) 
df_upsampled.insert(3, "Fuzzy", fuzzy, True) 
df_upsampled.insert(3, "Euclid", euclid, True) 
df_upsampled.insert(3, "Noun", noun, True)
df_upsampled.insert(3, "Adjective", adjective, True)
df_upsampled.insert(3, "Verb", verb, True)

X = df_upsampled.iloc[:,3:16]
y = df_upsampled.iloc[:, [16]]

scaler = MMS()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

log_reg = LR()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

print(SM.accuracy_score(y_test,y_pred))
print(SM.recall_score(y_test,y_pred))
print(SM.precision_score(y_test,y_pred))



