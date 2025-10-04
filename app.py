from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipelineimport
import uvicorn


class SensorData(BaseModel):
    x: float
    y: float
    z: float


app = FastAPI(title="Posture Predictor API")


pipeline = CNNLSTMPipelineimport()
pipeline.load_pipeline("pipeline.pkl")

@app.get("/")
def root():
    return {"status": "running", "message": "Posture Predictor API aktif"}

@app.post("/predict")
def predict(data: SensorData):
    """Tek bir (x, y, z) girdisiyle tahmin yapar"""
    try:
        posture, confidence = pipeline.add_data_point(data.x, data.y, data.z)
        
        
        if posture is None:
            return {
                "status": "collecting",
                "message": f"Veri toplanıyor... ({len(pipeline.data_buffer)}/{pipeline.window_size})"
            }
        
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": confidence
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
