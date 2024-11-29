from preprocessing import InputPreprocessor
import numpy as np
import pandas as pd
import joblib


def test_data_transform(question1, question2):
    test = pd.DataFrame(data={"col1": question1, "col2": question2}, index=[0])
    preprocessor = InputPreprocessor(test)
    preprocessor.remove_unimportant_data()
    preprocessor.remove_int_column()
    test = preprocessor.data

    test_vec = joblib.load("w2v.joblib")

    def document_vector(data):
        doc = [word for word in data.split() if word in test_vec.wv.index_to_key]
        if not doc:
            return np.zeros(test_vec.vector_size)
        return np.mean(test_vec.wv[doc], axis=0)

    X_test_w2v = []
    for column in test.columns:
        for doc in test[column].values:
            X_test_w2v.append(document_vector(doc))
    test = np.array(X_test_w2v)

    data_q1, data_q2 = np.vsplit(test, 2)

    temp1 = pd.DataFrame(data_q1)
    temp2 = pd.DataFrame(data_q2)
    test_idf = pd.concat([temp1, temp2], axis=1)

    pca = joblib.load("pca.joblib")
    test = pca.transform(test_idf)

    return test
