from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os


class SensorData(BaseModel):
    x: float
    y: float
    z: float

class SmoothingConfig(BaseModel):
    smoothing_factor: float

app = FastAPI(title="Real-time Posture API")
# allow each domains
origins = [
    "*",  # Geliştirme için serbest bırak
    "https://posture-api-krqg.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline yükleme
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline

pipeline = CNNLSTMPipeline(window_size=15)
try:
    pipeline.load_pipeline("pipeline")
    print("✅ Model yüklendi")
except Exception as e:
    if not os.path.exists("cnn_lstm_hybrid_model/datasets/new_dataset.csv"):
        raise RuntimeError("Dataset dosyası bulunamadı. Model eğitilemedi.")
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
    """Gerçek zamanlı tahmin - hızlı tepkili"""
    try:
        posture, confidence, all_predictions = pipeline.add_data_point(data.x, data.y, data.z)
        
        current_size = len(pipeline.data_buffer)
        
        if posture is None:
            return {
                "status": "collecting",
                "message": f"Kalibrasyon ({current_size}/{pipeline.window_size})",
                "buffer_size": current_size
            }
        
        # Kısmi mi tam mı tahmin?
        prediction_type = "partial" if current_size < pipeline.window_size else "full"
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": float(confidence),
            "buffer_size": len(pipeline.data_buffer),
            "prediction_type": prediction_type,
            "all_predictions": all_predictions
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
    