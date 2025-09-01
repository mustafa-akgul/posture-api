import pandas as pd
import fastapi as FastAPI
import tensorflow as tf
import numpy as np
from Posture_Model.cnn_lstm_hybrid_model import cnn_lstm_pipeline
app = FastAPI()
pipeline = cnn_lstm_pipeline.CNNLSTMPipeline()
pipeline.load_pipeline("cnn_lstm_model.h5")

@app.post("/predict")
def predict(data: list):
    df = pd.DataFrame(data)[['x','y','z']]

    predictions = pipeline.predict(df)

    return {"prediction": predictions.tolist()}