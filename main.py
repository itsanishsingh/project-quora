from sklearn.pipeline import Pipeline
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


def training_pipeline(data):
    reader = InputReader(data)
    df = reader.reader_logic()
    df = df.sample(5000, random_state=42)
    print("Read done")

    preprocessor = InputPreprocessor(df)
    preprocessor.drop_na()
    preprocessor.drop_duplicate()
    df = preprocessor.data
    print("Initial preprocess done")

    splitter = Splitter(df)
    X, y = splitter.target_split("is_duplicate")
    print("Split done")

    preprocessor = InputPreprocessor(X)
    preprocessor.remove_unimportant_data()
    preprocessor.remove_int_column()
    X = preprocessor.data
    print("Further processing done")

    X_train, X_test, y_train, y_test = splitter.split(X, y)
    print("Train test split done")

    train_vec = Vectorize(X_train)
    X_train = train_vec.tfidf()
    # train_vec.save()

    test_vec = joblib.load("tfidf.joblib")

    data_combined = []
    for _, val in X_test.items():
        temp = list(val)
        data_combined += temp

    test1, test2 = np.vsplit(test_vec.transform(data_combined).toarray(), 2)

    temp1 = pd.DataFrame(test1)
    temp2 = pd.DataFrame(test2)
    X_test = pd.concat([temp1, temp2], axis=1)
    print("Vectorization done")

    ext = FeatureManipulator(X_train, X_test)
    X_train, X_test = ext.extraction()
    # ext.save()
    print("PCA done")

    model = Modeller(X_train, y_train)
    model.rand_forest_clf()
    model.fit()
    y_pred = model.predict(X_test)

    print(accuracy_score(y_test, y_pred))
    # model.save()


def main():
    data = "data/questions.csv"
    training_pipeline(data)


if __name__ == "__main__":
    main()
