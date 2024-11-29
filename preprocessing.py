from nltk.corpus import stopwords
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
import re
import string


class InputPreprocessor:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def impute_mean(data):
        imp = SimpleImputer()
        return pd.Series(
            imp.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index
        )

    @staticmethod
    def impute_mode(data):
        imp = SimpleImputer(strategy="most_frequent")
        return pd.Series(
            imp.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index
        )

    def imputer(self):
        print("Before imputing: ")
        print(self.data.isnull().sum())
        for column in self.data.columns:
            if self.data[column].dtype in ["object"]:
                self.data[column] = self.impute_mode(self.data[column])
            elif self.data[column].dtype in ["float64", "int64"]:
                self.data[column] = self.impute_mean(self.data[column])
        print("After imputing: ")
        print(self.data.isnull().sum())

    def drop_na(self):
        self.data = self.data.dropna()

    def drop_duplicate(self):
        print("Before dropping duplicate: ")
        print(self.data.duplicated().sum())
        self.data = self.data.drop_duplicates()
        print("After dropping duplicate: ")
        print(self.data.duplicated().sum())

    @staticmethod
    def remove_html(data):
        pattern = re.compile("<.*?>")
        return pattern.sub(r"", data)

    @staticmethod
    def remove_url(data):
        pattern = re.compile(r"https?://\S+|www\.\S+")
        return pattern.sub(r"", data)

    @staticmethod
    def remove_punctuation(data):
        return data.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def remove_stopwords(data):
        en_stopwords = stopwords.words("english")
        end_string = []
        for word in data.split():
            if word not in en_stopwords:
                end_string.append(word)
        return " ".join(end_string)

    def remove_unimportant_data(self):
        categorical_features = self.data.select_dtypes(include=["object"]).columns
        for col in categorical_features:
            self.data[col] = self.data[col].str.lower()
            self.data[col] = self.data[col].apply(self.remove_html)
            self.data[col] = self.data[col].apply(self.remove_url)
            self.data[col] = self.data[col].apply(self.remove_punctuation)
            self.data[col] = self.data[col].apply(self.remove_stopwords)

    def remove_int_column(self):
        numerical_features = self.data.select_dtypes(
            include=["float64", "int64"]
        ).columns

        self.data = self.data.drop(numerical_features, axis=1)
