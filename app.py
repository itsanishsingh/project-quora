from fastapi import FastAPI, Request
import joblib
from testing_pipeline import test_data_transform
from database_handler import DBHandler


def prediction(question_1, question_2):
    data = test_data_transform(question_1, question_2)
    model = joblib.load("model_w2v.joblib")
    response = model.predict(data)[0]

    db = DBHandler()
    db.connect()
    # db.create_table()
    db_dict = {
        "question1": question_1,
        "question2": question_2,
        "prediction": response,
    }
    db.upload(db_dict)

    return str(response)


app = FastAPI()


@app.post("/predict")
async def read_root(request: Request):
    try:
        data = await request.json()
        question_1 = data["question1"]
        question_2 = data["question2"]

        response = prediction(question_1, question_2)

        return {"Result": response}

    except Exception as e:
        return {"Error": str(e)}
