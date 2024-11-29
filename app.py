from fastapi import FastAPI, Request
import joblib
from testing_pipeline import test_data_transform


def prediction(question_1, question_2):
    data = test_data_transform(question_1, question_2)
    model = joblib.load("model_w2v.joblib")
    response = model.predict(data)

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
