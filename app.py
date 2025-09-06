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
    try:
        df = pd.DataFrame(input_data.data)[['x','y','z']]

        if len(df) < pipeline.window_size:
            return{
                "error": f"Min {pipeline.window_size} line need, now, there are {len(df)} line. ",
                "prediction": []
            }
        predictions = pipeline.predict(df)
        return {"prediction": predictions.tolist()}
    
    except Exception as e:
        return {"error": str(e), "received_data": input_data.data}

        
        
