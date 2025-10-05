from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn

class SensorData(BaseModel):
    x: float
    y: float
    z: float

class SmoothingConfig(BaseModel):
    smoothing_factor: float

app = FastAPI(title="Real-time Posture API")

# Pipeline yükleme
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline

pipeline = CNNLSTMPipeline(window_size=15)
try:
    pipeline.load_pipeline("pipeline")
    print("✅ Model yüklendi")
except Exception as e:
    print(f"⚠️ Model yüklenemedi: {e}")
    print("Yeniden eğitiliyor...")
    df = pd.read_csv("cnn_lstm_hybrid_model/datasets/new_dataset.csv")
    pipeline.fit(df, epochs=30, batch_size=32)
    pipeline.save_pipeline("pipeline")
    print("✅ Model eğitildi ve kaydedildi")

@app.get("/")
def root():
    return {
        "status": "running",
        "pipeline_stats": pipeline.get_prediction_stats()
    }

@app.post("/predict")
def predict(data: SensorData):
    """Gerçek zamanlı ağırlıklı tahmin"""
    try:
        posture, confidence, all_predictions = pipeline.add_data_point(data.x, data.y, data.z)
        
        if posture is None:
            return {
                "status": "collecting",
                "message": f"Kalibrasyon ({len(pipeline.data_buffer)}/{pipeline.window_size})",
                "buffer_size": len(pipeline.data_buffer)
            }
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": confidence,
            "all_predictions": all_predictions,
            "prediction_type": "temporal_weighted_smoothed"
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_smoothing")
def update_smoothing(config: SmoothingConfig):
    """Yumuşatma faktörünü güncelle"""
    pipeline.set_smoothing_factor(config.smoothing_factor)
    return {
        "status": "success", 
        "smoothing_factor": pipeline.smoothing_factor
    }

@app.get("/pipeline_stats")
def get_pipeline_stats():
    """Pipeline istatistiklerini getir"""
    return pipeline.get_prediction_stats()

@app.post("/reset")
def reset_buffer():
    """Buffer'ı sıfırla"""
    pipeline.reset_buffer()
    return {"status": "success", "message": "Pipeline buffer sıfırlandı"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)