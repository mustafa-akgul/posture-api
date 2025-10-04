from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline
import uvicorn


class SensorData(BaseModel):
    x: float
    y: float
    z: float


app = FastAPI(title="Posture Predictor API")


pipeline = CNNLSTMPipeline()
try:
    pipeline.load_pipeline("pipeline")
    print("Model yüklendi")
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    print("Yeniden eğitiliyor...")
    df = pd.read_csv("cnn_lstm_hybrid_model/datasets/new_dataset.csv")
    pipeline.fit(df, epochs=30, batch_size=32)
    pipeline.save_pipeline("pipeline")
    print("Model eğitildi ve kaydedildi")


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
