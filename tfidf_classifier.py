from joblib import load
import re
from nltk.tokenize import RegexpTokenizer
from spacy.lang.ru.stop_words import STOP_WORDS
from nltk.tokenize import WordPunctTokenizer
from pymystem3 import Mystem


class Tfidf_classifier():
    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.word_punct_tokenizer = WordPunctTokenizer()
        self.mystem = Mystem()
        self.classifier = load('saved_sklearn_classifier/tfidf_classifier.joblib')

    def clean(self, sent):
        sent = sent[0]
        sent = sent.lower()  # lower string
        sent = re.sub(r'\d+', '', sent)  # remove numbers
        sent = ' '.join(self.tokenizer.tokenize(sent))  # remove punctuation
        sent = [word for word in self.word_punct_tokenizer.tokenize(sent) if not word in STOP_WORDS]  # remove stop words
        sent = ' '.join([self.mystem.lemmatize(word)[0] for word in sent])
        return sent

    def predict(self, sent):
        cleaned_sent = self.clean(sent)
        prediction = self.classifier.predict([cleaned_sent])
        if prediction[0] == 1.0:
            return 'позитивный отзыв'
        else:
            return 'негативный отзыв'
