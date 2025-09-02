import pandas as pd
from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline

app = FastAPI()
pipeline = CNNLSTMPipeline()
pipeline.load_pipeline("pipeline.pkl")

@app.post("/predict")
def predict(data: list):
    df = pd.DataFrame(data)[['x','y','z']]

    predictions = pipeline.predict(df)

    return {"prediction": predictions.tolist()}