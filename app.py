from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/predict")
async def read_root(request: Request):
    try:
        data = await request.json()
        question_1 = data["question1"]
        question_2 = data["question2"]

        return {"Q1": question_1, "Q2": question_2}

    except Exception as e:
        return {"Error": e}
