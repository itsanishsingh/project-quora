import gensim
from gensim.utils import simple_preprocess
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

import numpy as np
import pandas as pd


class Vectorize:
    def __init__(self, data):
        self.data = data

    def tfidf(self):
        self.tfidf = TfidfVectorizer()

        data_combined = []
        for _, val in self.data.items():
            temp = list(val)
            data_combined += temp

        data_q1, data_q2 = np.vsplit(
            self.tfidf.fit_transform(data_combined).toarray(), 2
        )

        temp1 = pd.DataFrame(data_q1)
        temp2 = pd.DataFrame(data_q2)
        data_idf = pd.concat([temp1, temp2], axis=1)

        return data_idf

    def save(self):
        joblib.dump(self.tfidf, "tfidf.joblib")

    def create_story(self):
        story = []
        for column in self.data.columns:
            for word in self.data[column]:
                raw_sent = sent_tokenize(word)
                for sent in raw_sent:
                    story.append(simple_preprocess(sent))

        return story

    def create_w2v_model(self):
        self.model = gensim.models.Word2Vec(
            window=10,
            min_count=2,
        )

        story = self.create_story()
        self.model.build_vocab(story)
        self.model.train(
            story, total_examples=self.model.corpus_count, epochs=self.model.epochs
        )

    def save_w2v(self):
        joblib.dump(self.model, "w2v.joblib")
