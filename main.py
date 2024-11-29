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

    train_w2v = Vectorize(X_train)
    train_w2v.create_w2v_model()
    # train_w2v.save_w2v()
    w2v_model = joblib.load("w2v.joblib")

    def document_vector(data):
        doc = [word for word in data.split() if word in w2v_model.wv.index_to_key]
        if not doc:
            return np.zeros(w2v_model.vector_size)
        return np.mean(w2v_model.wv[doc], axis=0)

    X_train_w2v = []
    for column in X_train.columns:
        for doc in X_train[column].values:
            X_train_w2v.append(document_vector(doc))
    X_train = np.array(X_train_w2v)

    X_test_w2v = []
    for column in X_test.columns:
        for doc in X_test[column].values:
            X_test_w2v.append(document_vector(doc))
    X_test = np.array(X_test_w2v)

    print("Vectorization done")

    train1, train2 = np.vsplit(X_train, 2)
    temp1 = pd.DataFrame(train1)
    temp2 = pd.DataFrame(train2)
    X_train = pd.concat([temp1, temp2], axis=1)

    test1, test2 = np.vsplit(X_test, 2)
    temp1 = pd.DataFrame(test1)
    temp2 = pd.DataFrame(test2)
    X_test = pd.concat([temp1, temp2], axis=1)

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
