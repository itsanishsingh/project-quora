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

    test_vec = joblib.load("tfidf.joblib")

    data_combined = [test.iloc[0, 0], test.iloc[0, 1]]

    data_q1, data_q2 = np.vsplit(test_vec.transform(data_combined).toarray(), 2)

    temp1 = pd.DataFrame(data_q1)
    temp2 = pd.DataFrame(data_q2)
    test_idf = pd.concat([temp1, temp2], axis=1)

    pca = joblib.load("pca.joblib")
    test = pca.transform(test_idf)

    return test
