from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline

class InputData(BaseModel):
    data: list

app = FastAPI()
pipeline = CNNLSTMPipeline()
pipeline.load_pipeline("pipeline.pkl")

@app.post("/predict")
def predict(input_data: InputData):
    df = pd.DataFrame(input_data.data)[['x','y','z']]
    predictions = pipeline.predict(df)
    return {"prediction": predictions.tolist()}
