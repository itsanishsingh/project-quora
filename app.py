from fastapi import FastAPI, Request

app = FastAPI()


@app.post("/predict")
async def read_root(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        return {"Error": e}
    else:
        return data
