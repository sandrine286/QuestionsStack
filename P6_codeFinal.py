import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.stem.snowball import EnglishStemmer
import re
import pickle

import argparse

def corpus_tokenizer(corpora):
    # pour écupérer les valeurs alphanumériques ASCII
    tokenizer = nltk.RegexpTokenizer(r'[a-z]+', flags=re.ASCII)
    stemmer = EnglishStemmer()
    stems = []
    # suppresion des expressions régulières, passage en minuscule
    tokens = tokenizer.tokenize(corpora.lower())
    # racinification des mots
    stems += [stemmer.stem(w) for w in tokens]
    return stems

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", help="""Texte de la question Stack Overflow à passer""")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    question = args.text
    print("Vous avez saisi :\n", question)

    ###
    # Chargement des fichiers données + modèle + fonction de normalisation
    ###

    vect = pickle.load(open('data/vectorizer.pkl', 'rb'))
    tdidf = pickle.load(open('data/tdidf.pkl', 'rb'))
    lda = pickle.load(open('data/lda18.pkl', 'rb'))

    ###
    # création de la matrice creuse tdidf et prédiction du modèle LDA
    ###
    X_test = vect.transform(pd.DataFrame(data=[str],columns=['corpus']))
    X_test_tfidf = tdidf.transform(X_test)
    res = lda.transform(X_test_tfidf)
    max_index_col = np.argmax(res)+1
    print("La question appartient au Topic", max_index_col)