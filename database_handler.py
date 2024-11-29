from sqlalchemy import create_engine, text
from sensitive import *


class DBHandler:
    def __init__(self):
        pass

    def connect(self):
        self.engine = create_engine(postgres_url)

    def create_table(self):
        query = """create table quora_table(
        id SERIAL Primary key,
        question1 text,
        question2 text,
        prediction int)"""
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                conn.execute(text(query))
                trans.commit()

    def upload(self, data):
        q1 = data["question1"]
        q2 = data["question2"]
        pred = data["prediction"]
        query = f"insert into quora_table(question1, question2, prediction) values('{q1}', '{q2}', {pred});"
        with self.engine.connect() as conn:
            with conn.begin() as trans:
                conn.execute(text(query))
                trans.commit()
