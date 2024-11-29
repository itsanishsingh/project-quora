from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import joblib

from input_reading import InputReader
from preprocessing import InputPreprocessor
from splitting import Splitter
from text_vectorization import Vectorize
from feature_engineering import FeatureManipulator
from modelling import Modeller


def training_pipeline(data, target):
    reader = InputReader(data)
    df = reader.reader_logic()
    df = df.sample(5000, random_state=42)
    print("Read done")
    print("Description of data: ", df.describe())

    preprocessor = InputPreprocessor(df)
    preprocessor.imputer()
    preprocessor.drop_duplicate()
    df = preprocessor.data
    print("Initial preprocess done")

    splitter = Splitter(df)
    X, y = splitter.target_split(target)
    print("Split done")

    preprocessor = InputPreprocessor(X)
    preprocessor.remove_unimportant_data()
    preprocessor.remove_int_column()
    X = preprocessor.data
    print("Further processing done")

    X_train, X_test, y_train, y_test = splitter.split(X, y)
    print("Train test split done")

    train_w2v = Vectorize(X_train)
    train_w2v.create_w2v_model()
    X_train = train_w2v.vectorize(X_train)
    X_test = train_w2v.vectorize(X_test)
    print("Vectorization done")

    ext = FeatureManipulator(X_train, X_test)
    X_train, X_test = ext.extraction()
    print("PCA done")

    model = Modeller(X_train, y_train)
    model.rand_forest_clf()
    model.fit()
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))

    # train_w2v.save_w2v()
    # ext.save()
    # model.save()


def main():
    data = "data/questions.csv"
    target = "is_duplicate"
    training_pipeline(data, target)


if __name__ == "__main__":
    main()
