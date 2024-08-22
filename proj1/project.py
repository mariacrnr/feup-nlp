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
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
import json
import sys
import time
from sklearn.feature_selection import SelectKBest, chi2


import numpy as np

class MyEmbedder:
    def __init__(self, embeddings, sequence_length, operation):
        self.embeddings = embeddings
        self.sequence_length = sequence_length
        self.operation = operation

    def text_to_mean_vector(self,embeddings, corpus):
        split_corpus = []
        for sent in corpus:
            split_corpus.append(sent.split())
        vec = embeddings.sents2elmo(split_corpus)
        
        res = []
        for sent in vec:
            res.append(np.mean(sent, axis=0))
        return res
    
    def text_to_mean_std_vector(self, embeddings, corpus):
        split_corpus = []
        for sent in corpus:
            split_corpus.append(sent.split())
        vec = embeddings.sents2elmo(split_corpus)
        
        res = []
        for sent in vec:
            res.append(np.append(np.mean(sent, axis=0), [np.std(sent, axis=0)]))
        return res

    def tokens_to_vector(self,embeddings, corpus, sequence_len):
        # convert tokens to embedding vectors, up to sequence_len tokens
        split_corpus = []
        for sent in corpus:
            split_corpus.append(sent.split())
        vec = embeddings.sents2elmo(split_corpus, output_layer=0)
        res = []
        for emb in vec:
            if(len(emb) >= sequence_len):
                emb = emb[:sequence_len]
            else:
                emb = np.append(emb, np.zeros(1024*(sequence_len - len(emb))))
            res.append(emb.flatten())
    
        return res

    def fit_transform(self, corpus):
        features = []
        for i in range(0, len(corpus), 25):
            print("Getting embeddings on :", i)
            if self.operation == "concat":
                features += self.tokens_to_vector(self.embeddings, corpus[i:i+25], self.sequence_length)
            elif self.operation == "mean":
                features += self.text_to_mean_vector(self.embeddings, corpus[i:i+25])
            elif self.operation == "std":
                features += self.text_to_mean_std_vector(self.embeddings, corpus[i:i+25])
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
    sw = set(stopwords.words('french' if lang=='fr' else 'german'))
    corpus = []
    
    for i in range(0, data['content'].size):
        text = regex.sub(r'<U\+([0-9a-fA-F]+)>', lambda m: chr(int(m.group(1),16)), data['content'][i])
        text = emoji.demojize(text, language=lang)
        # get review and remove non alpha chars
        text = regex.sub(reg, ' ', text)
        text = text.lower()
        # split into tokens, apply stemming and remove stop words
        text = ' '.join([t.lemma_ for t in nlp(text)])
        corpus.append(text)

    return corpus


def train_model(estimator, X_train, y_train):
    if estimator == "svc":
        clf = SVC()
        param_grid = {
            'C' : [0.1, 1, 10, 100], 
            'gamma' : ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'probability': [True]
        }

    elif estimator == "lr":
        clf = LogisticRegression()
        param_grid = {
            'C' : [1], 
            'penalty' : ['l2'],
            'solver': ['liblinear']
        }

    elif estimator == "sgd":
        clf = SGDClassifier()
        param_grid = {
            'loss': ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'eta0': [0.03, 0.01, 0.003, 0.001, 0.0003],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'alpha': [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003],
            'fit_intercept': [True, False],
            'shuffle': [True, False]
        }

    elif estimator == "dt":
        clf = DecisionTreeClassifier()
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [2, 4, 6, 8, 10, 12, None],
            'max_features': [12, 13, 14, 15, 16],
            'class_weight' : [None] # weights
        }

    elif estimator == "mnb":
        clf = MultinomialNB()
        param_grid = {
            
        }
    
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search


def main():
    config_name = sys.argv[1]

    configf = open(f'./model_configs/{config_name}.json')
    config = json.load(configf)
    
    print("Using dataset " + config["dataset"])
    
    print("Applying preprocessing..")
    
    if config["dataset"] == "fr":
        corpus = processor(fr_train, 'fr', nlp_fr) + processor(fr_val, 'fr', nlp_fr) + processor(fr_test, 'fr', nlp_fr)
        train = fr_train
        val = fr_val
        test = fr_test
    else:
        corpus = processor(de_train, 'de', nlp_de) + processor(de_val, 'de', nlp_de) + processor(de_test, 'de', nlp_de)
        train = de_train
        val = de_val
        test = de_test
    

    print(len(corpus))

    start = time.time()

    print("Applying Vectorizer " + config["featureExtraction"]["vectorizer"] + "..")

    ngrams_range = (config["featureExtraction"]["ngrams_low"],config["featureExtraction"]["ngrams_upp"])

    if config["featureExtraction"]["vectorizer"] == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=ngrams_range)
    elif config["featureExtraction"]["vectorizer"] == "count":
        vectorizer = CountVectorizer(ngram_range=ngrams_range)
    elif config["featureExtraction"]["vectorizer"] == "embeddings":
        embeddings = Embedder(f'../word_embeddings/{config["dataset"]}_model')
        vectorizer = MyEmbedder(embeddings, 55 if config["dataset"] == "de" else 40, config["featureExtraction"]["agg"])

    X = vectorizer.fit_transform(corpus)

    if config["featureExtraction"]["vectorizer"] != "embeddings":
        X = X.toarray()
    else:
        X = np.array(X)

    X = pd.DataFrame(X)
    X["id"] = pd.concat([train["id"], val["id"], test["id"]]).to_numpy()

    print(X)
    print(X.shape)

    target = config["featureExtraction"]["target"]
    y = np.concatenate((train[target].values, val[target].values, test[target].values))
    print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    ids = X_test["id"].values

    X_train.drop(columns=["id"], inplace=True)
    X_test.drop(columns=["id"], inplace=True)

    if (config["featureExtraction"]["vectorizer"] == "embeddings" and config["featureExtraction"]["agg"] == "concat"):
        #chi2 is only for non-negative numbers need scaling
        scaler = MinMaxScaler()
        scaler.fit_transform(X_train)
        scaler.transform(X_test)

        selector = SelectKBest(chi2, k=30000)
        selector.fit_transform(X_train, y_train)
        selector.transform(X_test)
    
    if (config["featureExtraction"]["vectorizer"] == "tfidf" and config["featureExtraction"]["ngrams_upp"] >= 2):
        selector = SelectKBest(chi2, k=30000)
        selector.fit_transform(X_train, y_train)
        selector.transform(X_test)

    kbest = SelectKBest(chi2, k=30000)
    X_train = kbest.fit_transform(X_train, y_train)

    if config["featureExtraction"]["oversampling"]:
        oversample = SMOTE()
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    
    grid_search = train_model(config["classification"]["estimator"], X_train, y_train)
    
    end = time.time()
    
    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
    print('Best estimator: {}'.format(grid_search.best_estimator_))
    
    y_pred = grid_search.best_estimator_.predict(kbest.transform(X_test))
    print(accuracy_score(y_test, y_pred))

    new_f1_score = f1_score(y_test, y_pred, average='macro')
    new_accuracy_score = accuracy_score(y_test, y_pred)

    misclassified_cases = []

    for i in range(len(y_test)):
        if y_pred[i] != y_test[i]:
            misclassified_cases.append(
                {
                    "case": int(ids[i]),
                    "predicted": int(y_pred[i]),
                    "expected": int(y_test[i])
                }
            )

    train_result = {
        "start": start,
        "end": end,
        "best_parameters": grid_search.best_params_,
        "f1": new_f1_score,
        "accuracy": new_accuracy_score,
        "misclassified_cases": misclassified_cases
    }
    print("F1:" , new_f1_score)
    config["results"] = train_result

    with open(f"./model_configs/{config_name}.json", 'w') as outfile:
        json.dump(config, outfile, indent=2)
    
    
if __name__ == "__main__":
    main()