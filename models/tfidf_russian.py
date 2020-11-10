import pandas as pd

data_pos = pd.read_csv('positive_reviews.csv')
data_neg = pd.read_csv('negative_revews.csv')
#%%
import numpy as np

pos_data = data_pos.iloc[:, 1].values
neg_data = data_neg.iloc[:, 1].values

#oversampling
neg_data = np.repeat(neg_data, 8)
print('shapes of pos and neg data: ', pos_data.shape, neg_data.shape)

#final data
data = np.concatenate((pos_data, neg_data))
print('shape of final data: ', data.shape)

data = np.array([' '.join(sent.split()[:195]) for sent in data])

#targets
pos_targets = np.ones(shape=(8786, ))
neg_targets = np.zeros(shape=(8360, ))

targets = np.concatenate((pos_targets, neg_targets))
print('targets shape: ', targets.shape)
#%%
#text cleaning

import re
from nltk.tokenize import RegexpTokenizer
from spacy.lang.ru.stop_words import STOP_WORDS
from nltk.tokenize import WordPunctTokenizer
from pymystem3 import Mystem


tokenizer = RegexpTokenizer(r'\w+')
word_punct_tokenizer = WordPunctTokenizer()
mystem = Mystem()


def clean_text(data):
    cleaned_sent = []
    for sent in data:
        sent = sent.lower() #lower string
        sent = re.sub(r'\d+', '', sent) #remove numbers
        sent = ' '.join(tokenizer.tokenize(sent)) #remove punctuation
        sent = [word for word in word_punct_tokenizer.tokenize(sent) if not word in STOP_WORDS]  #remove stop words
        sent = ' '.join([mystem.lemmatize(word)[0] for word in sent]) #lemmatize
        cleaned_sent.append(sent)
    return cleaned_sent


data = clean_text(data)
#%%
print(data[:30])

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection


classifier = Pipeline([('vectorizer', TfidfVectorizer(analyzer='char_wb')), ('classifier', LogisticRegression())])

print(classifier.get_params().keys())

parameters_grid = {
    'vectorizer__ngram_range': [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10)],
    'vectorizer__max_df': [0.5, 0.6, 0.7, 0.8],
    'vectorizer__min_df': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
}


cv = model_selection.StratifiedShuffleSplit(n_splits=5, test_size = 0.2)

randomized_grid_cv = model_selection.RandomizedSearchCV(classifier, parameters_grid, scoring='accuracy', cv=cv,
                                                                                                n_iter=15)

cv_data = np.concatenate((data[:500], data[-500:]))
cv_targets = np.concatenate((np.ones(shape=(500, )), np.zeros(shape=(500, ))))
print(cv_data.shape)
print(cv_targets.shape)

randomized_grid_cv.fit(cv_data, cv_targets)
#%%
print(randomized_grid_cv.best_params_)
print(randomized_grid_cv.best_score_) #0.909
#%%
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=0.1)
print(type(data_train), type(targets_train))

classifier_final = Pipeline([('vectorizer', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3),
                                                      min_df=0.05, max_df=0.5)), ('classifier', LogisticRegression())])

classifier_final.fit(data_train, targets_train)

#%%
predictions = classifier_final.predict(data_test)

#0.8513119533527697 0.8509643483343075
print(accuracy_score(targets_test, predictions), f1_score(targets_test, predictions))
#%%
from joblib import dump, load

dump(classifier_final, 'saved_sklearn_classifier/tfidf_classifier.joblib')
#%%
sent = clean_text(['Я люблю этот телефон'])
print(sent)
print(classifier_final.predict(sent))

