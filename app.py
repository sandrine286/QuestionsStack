# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.stem.snowball import EnglishStemmer
import re
import pickle


##############################################
# fonction pour la tokenisation du text
##############################################
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


app = Flask(__name__)

#######################
# APPLICATION PPALE
#######################


@app.route("/")
def hello():
    return "Le chemin de 'racine' est : " + request.path


###
# Chargement des fichiers données + modèle + fonction de normalisation
###
vect = pickle.load(open('data/vectorizerComplet.pkl', 'rb'))
tdidf = pickle.load(open('data/tdidf.pkl', 'rb'))
reg = pickle.load(open('data/supervise_model.sav', 'rb'))
listLabel = pd.read_csv("data/listLabel.csv", sep=";")
lda = pickle.load(open('data/lda_model.sav', 'rb'))

#############################
## APPLICATION - FORMULAIRE
#############################
@app.route('/questions/', methods=['GET', 'POST'])
def questions():
    print(request.method)
    msg = ""

    if request.method == 'POST':
        model = request.form['model']
        print("p1 :", request.form['model'])
        print("p2 :", request.form['question'])
        question = request.form['question']
        if len(question) == 0:
            msg = "Champ obligatoire."
        X_test = vect.transform(pd.Series([question]))
        X_test_tfidf = tdidf.transform(X_test)

        ###
        # test modèle
        ###
        if model == 'REG':
            y_pred = reg.predict(X_test_tfidf)
            tag_name = listLabel[listLabel["target_id"]
                                 == y_pred[0]]['target_name'].values
            print("Le tag de la question est ", tag_name,y_pred[0])
            tag = "Le tag de la question est " + str(tag_name[0])
        else:
            res = lda.transform(X_test_tfidf)
            print(res)

            max_index_col = np.argmax(res)+1
            print("La question appartient au Topic", max_index_col)
            tag = "La question appartient au Topic " + str(max_index_col)

        return render_template('index.html', msg=msg, model=model, question=question, tag=tag)
    else:
        tag = ""
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)