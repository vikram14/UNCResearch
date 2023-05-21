from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.sparse import csr_matrix
import re

  
lemmatizer = WordNetLemmatizer()
pca = PCA(n_components=2)
verbs_adverbs=set(['VB','VBP','VBG','VBD','VBZ','RBR'])
nouns_rest=set(['NN','NNP','FW','NNS','CD','CC','IN'])

def jaccardStringSimilarity(a,b):
    a=re.sub(r'#.*','',a)
    b=re.sub(r'#.*','',b)
    a= a.replace('.',' . ')
    b= b.replace('.',' . ')
    a_tok= word_tokenize(a.lower())
    b_tok = word_tokenize(b.lower())
    a2=[e[0] for e in list(filter(lambda tup: tup[1] in nouns_rest or tup[1]=='JJ' or tup[1] in verbs_adverbs , pos_tag(a_tok)))]
    b2= [e[0] for e in list(filter(lambda tup: tup[1] in nouns_rest or tup[1]=='JJ' or tup[1] in verbs_adverbs, pos_tag(b_tok)))]
    a1= [lemmatizer.lemmatize(w) for w in a2 if w not in string.punctuation]
    b1= [lemmatizer.lemmatize(w) for w in b2 if w not in string.punctuation]
    set_a = set(a1)
    set_b = set(b1)
    print(set_a, set_b)
    jaccard = len(set_a.intersection(set_b))/len(set_a.union(set_b))
    return 1- jaccard


def tfidfMatrix(methods):
    def preprocess(a):
        return " ".join( [lemmatizer.lemmatize(w) for w in word_tokenize(a) if w not in string.punctuation])
    corpora = [preprocess(method['description']) for method in methods]
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(corpora)
    return pca.fit_transform(csr_matrix.toarray(X))
