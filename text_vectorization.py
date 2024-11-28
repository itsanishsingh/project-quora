import gensim
from gensim.utils import simple_preprocess
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd


class Vectorize:
    def __init__(self, train, test):
        self.train = train
        self.test = test

    def tfidf(self):
        tfidf = TfidfVectorizer()

        train_combined = []
        for _, val in self.train.items():
            temp = list(val)
            train_combined += temp

        test_combined = []
        for _, val in self.test.items():
            temp = list(val)
            test_combined += temp

        train_q1, train_q2 = np.vsplit(tfidf.fit_transform(train_combined).toarray(), 2)
        test_q1, test_q2 = np.vsplit(tfidf.transform(test_combined).toarray(), 2)

        temp1 = pd.DataFrame(train_q1)
        temp2 = pd.DataFrame(train_q2)
        train_idf = pd.concat([temp1, temp2], axis=1)

        temp1 = pd.DataFrame(test_q1)
        temp2 = pd.DataFrame(test_q2)
        test_idf = pd.concat([temp1, temp2], axis=1)

        return train_idf, test_idf

    def create_story(self):
        story = []
        for word in self.data:
            raw_sent = sent_tokenize(word)
            for sent in raw_sent:
                story.append(simple_preprocess(sent))

        return story

    def vectorization(self):
        model = gensim.models.Word2Vec(
            window=10,
            min_count=2,
        )

        story = self.create_story()
        model.build_vocab(story)
        model.train(story, total_examples=model.corpus_count, epochs=model.epochs)
