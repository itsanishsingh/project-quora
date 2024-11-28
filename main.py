from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from input_reading import InputReader
from preprocessing import InputPreprocessor
from splitting import Splitter
from text_vectorization import Vectorize
from modelling import Modeller


def training_pipeline(data):
    reader = InputReader(data)
    df = reader.reader_logic()
    df = df.iloc[:10000]

    preprocessor = InputPreprocessor(df)
    preprocessor.drop_na()
    preprocessor.drop_duplicate()
    df = preprocessor.data

    splitter = Splitter(df)
    X, y = splitter.target_split("is_duplicate")

    preprocessor = InputPreprocessor(X)
    preprocessor.remove_unimportant_data()
    preprocessor.remove_int_column()
    X = preprocessor.data

    X_train, X_test, y_train, y_test = splitter.split(X, y)

    vec = Vectorize(X_train, X_test)
    X_train, X_test = vec.tfidf()

    print(X_train[0:10])
    print(X_test[0:10])

    # model = Modeller(X_train, y_train)
    # model.rand_forest_clf()
    # model.fit()
    # y_pred = model.predict(X_test)
    # print(accuracy_score(y_test, y_pred))


def main():
    data = "data/questions.csv"
    training_pipeline(data)


if __name__ == "__main__":
    main()
