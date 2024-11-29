from preprocessing import InputPreprocessor
import numpy as np
import pandas as pd
import joblib


def vectorize(model, data):
    def document_vector(data):
        doc = [word for word in data.split() if word in model.wv.index_to_key]
        if not doc:
            return np.zeros(model.vector_size)
        return np.mean(model.wv[doc], axis=0)

    temp_data = []
    for column in data.columns:
        for doc in data[column].values:
            temp_data.append(document_vector(doc))
    data = np.array(temp_data)

    temp1, temp2 = np.vsplit(data, 2)
    temp1 = pd.DataFrame(temp1)
    temp2 = pd.DataFrame(temp2)
    data = pd.concat([temp1, temp2], axis=1)

    return data


def test_data_transform(question1, question2):
    test = pd.DataFrame(data={"col1": question1, "col2": question2}, index=[0])
    preprocessor = InputPreprocessor(test)
    preprocessor.remove_unimportant_data()
    preprocessor.remove_int_column()
    test = preprocessor.data

    test_vec = joblib.load("w2v.joblib")

    test = vectorize(test_vec, test)

    return test
