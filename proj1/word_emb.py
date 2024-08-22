import pandas as pd
import numpy as np
import spacy
import regex
import emoji
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from elmoformanylangs import Embedder
import json
import sys

import numpy as np

class MyEmbedder:
    def __init__(self, embeddings, sequence_length, operation):
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.operation = operation

    def text_to_mean_vector(embeddings, text):
        tokens = text.split()
        
        # convert tokens to embedding vectors, up to sequence_len tokens
        vec = []
        i = 0
        while i < len(tokens):   # while there are tokens and did not reach desired sequence length
            try:
                vec.append(embeddings.get_vector(tokens[i]))
            except KeyError:
                True   # simply ignore out-of-vocabulary tokens
            finally:
                i += 1
        
        # add blanks up to sequence_len, if needed
        vec = np.mean(vec, axis=0)
        return vec

    def tokens_to_vector(self,embeddings, corpus, sequence_len):
        # convert tokens to embedding vectors, up to sequence_len tokens
        split_corpus = []
        for sent in corpus:
            split_corpus.append(sent.split())
        vec = embeddings.sents2elmo(split_corpus)
        res = []
        for sent in vec:
            res.append(sent.tolist())
        return res

    def fit_transform(self, corpus):
        features = []
        for i in range(0, len(corpus), 50):
            print("Getting embeddings on :", i)
            features += self.tokens_to_vector(self.embeddings, corpus[i:i+50], self.sequence_length)

        return features

## Load datasets ##
de_train = pd.read_csv("./train_dev_test_splits/de.train.csv", sep="\t")
de_val = pd.read_csv("./train_dev_test_splits/de.valid.csv", sep="\t")
de_test = pd.read_csv("./train_dev_test_splits/de.test.csv", sep="\t")

fr_train = pd.read_csv("./train_dev_test_splits/fr.train.csv", sep="\t")
fr_val = pd.read_csv("./train_dev_test_splits/fr.valid.csv", sep="\t")
fr_test = pd.read_csv("./train_dev_test_splits/fr.test.csv", sep="\t")


## Spacy Pipeline - Tokenization ##
nlp_de = spacy.load("de_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")

def processor(data, lang, nlp):
    reg = '[^a-zA-Z0-9 àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ]' if lang == 'fr' else '[^a-zA-Z0-9 äöüßÄÖÜẞ]'
    corpus = []
    for i in range(0, data['content'].size):
        text = regex.sub(r'<U\+([0-9a-fA-F]+)>', lambda m: chr(int(m.group(1),16)), data['content'][i])
        text = emoji.demojize(text, language=lang)
        # get review and remove non alpha chars
        text = regex.sub(reg, ' ', text)
        text = text.lower()
        # split into tokens, apply stemming and remove stop words
        text = ' '.join([t.text for t in nlp(text)])
        corpus.append(text)

    return corpus

def main():
    lang = sys.argv[1]

    if lang == "fr":
        corpus = processor(fr_train, 'fr', nlp_fr) + processor(fr_val, 'fr', nlp_fr) + processor(fr_test, 'fr', nlp_fr)
    else:
        corpus = processor(de_train, 'de', nlp_de) + processor(de_val, 'de', nlp_de) + processor(de_test, 'de', nlp_de)
        
    embeddings = Embedder(f'../word_embeddings/{lang}_model')
    vectorizer = MyEmbedder(embeddings, 80, "concat")

    X = vectorizer.fit_transform(corpus)

    with open(f"./embedds/{lang}.json", 'w') as outfile:
        json.dump({"embedd" : X}, outfile, indent=2)



if __name__ == "__main__":
    main()