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

class TransitionConfig(BaseModel):
    transition_threshold: float

app = FastAPI(title="Real-time Posture API v2 - Enhanced")

# CORS ayarları
origins = [
    "*",
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
    print("✅ Model yüklendi (Enhanced v2)")
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
    """API durumu ve pipeline istatistikleri"""
    stats = pipeline.get_prediction_stats()
    return {
        "status": "running",
        "version": "2.0-enhanced",
        "features": [
            "adaptive_smoothing",
            "posture_change_detection",
            "stability_tracking",
            "flicker_prevention"
        ],
        "pipeline_stats": stats
    }

@app.post("/predict")
def predict(data: SensorData):
    """
    Gerçek zamanlı tahmin - Gelişmiş versiyon
    - Otomatik duruş değişimi tespiti
    - Adaptive smoothing
    - Stabilite takibi
    """
    try:
        posture, confidence, all_predictions = pipeline.add_data_point(
            data.x, data.y, data.z
        )
        
        current_size = len(pipeline.data_buffer)
        
        # Hala kalibrasyon aşamasında
        if posture is None:
            return {
                "status": "collecting",
                "message": f"Kalibrasyon ({current_size}/{pipeline.window_size})",
                "buffer_size": current_size,
                "progress_percentage": round((current_size / pipeline.window_size) * 100, 1)
            }
        
        # Tam tahmin
        stats = pipeline.get_prediction_stats()
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": round(float(confidence), 3),
            "buffer_size": current_size,
            "all_predictions": all_predictions,
            
            # YENİ: Gelişmiş metrikler
            "stability": {
                "stable_count": stats["stable_count"],
                "is_stable": stats["stable_count"] >= 2,
                "last_prediction": stats["last_prediction"]
            },
            "smoothing_info": {
                "current_factor": stats["smoothing_factor"],
                "history_size": stats["prediction_history_size"]
            }
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_smoothing")
def update_smoothing(config: SmoothingConfig):
    """
    Yumuşatma faktörünü güncelle
    - 0.0: Çok smooth (yavaş değişim)
    - 1.0: Çok reaktif (hızlı değişim)
    - Önerilen: 0.6-0.8 arası
    """
    if not 0.0 <= config.smoothing_factor <= 1.0:
        raise HTTPException(
            status_code=400, 
            detail="Smoothing factor 0.0 ile 1.0 arasında olmalı"
        )
    
    pipeline.set_smoothing_factor(config.smoothing_factor)
    return {
        "status": "success", 
        "smoothing_factor": pipeline.smoothing_factor,
        "description": f"{'Hızlı tepki' if config.smoothing_factor > 0.7 else 'Stabil tahmin'}"
    }

@app.post("/update_transition_threshold")
def update_transition_threshold(config: TransitionConfig):
    """
    Duruş değişimi tespit eşiğini güncelle
    - Düşük değer: Daha hassas (küçük hareketlerde değişim)
    - Yüksek değer: Daha az hassas (büyük hareketlerde değişim)
    - Önerilen: 0.3-0.5 arası
    """
    if not 0.0 <= config.transition_threshold <= 1.0:
        raise HTTPException(
            status_code=400, 
            detail="Transition threshold 0.0 ile 1.0 arasında olmalı"
        )
    
    pipeline.set_transition_threshold(config.transition_threshold)
    return {
        "status": "success", 
        "transition_threshold": pipeline.transition_threshold,
        "description": f"{'Hassas' if config.transition_threshold < 0.4 else 'Normal'} tespit"
    }

@app.get("/pipeline_stats")
def get_pipeline_stats():
    """Detaylı pipeline istatistikleri"""
    stats = pipeline.get_prediction_stats()
    
    # Buffer doluluk yüzdesi
    buffer_percentage = (stats["buffer_size"] / stats["window_size"]) * 100
    
    # Stabilite durumu
    stability_status = "stable" if stats["stable_count"] >= 2 else "transitioning"
    
    return {
        **stats,
        "buffer_percentage": round(buffer_percentage, 1),
        "stability_status": stability_status,
        "is_ready": stats["buffer_size"] >= stats["window_size"]
    }

@app.post("/reset")
def reset_buffer():
    """
    Pipeline buffer'ını tamamen sıfırla
    - Tüm geçmiş verileri temizler
    - Yeni kalibrasyon gerektirir
    - Duruş değişimi sonrası kullanışlı
    """
    pipeline.reset_buffer()
    return {
        "status": "success", 
        "message": "Pipeline buffer sıfırlandı",
        "action_required": "Yeni veri göndermeye başlayın"
    }

@app.get("/health")
def health_check():
    """Sistem sağlık kontrolü"""
    stats = pipeline.get_prediction_stats()
    
    return {
        "status": "healthy" if pipeline.model is not None else "unhealthy",
        "model_loaded": pipeline.model is not None,
        "buffer_ready": stats["buffer_size"] >= stats["window_size"],
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/config")
def get_config():
    """Mevcut konfigürasyonu getir"""
    stats = pipeline.get_prediction_stats()
    
    return {
        "window_size": stats["window_size"],
        "smoothing_factor": stats["smoothing_factor"],
        "transition_threshold": getattr(pipeline, 'transition_threshold', 0.3),
        "prediction_history_size": stats["prediction_history_size"],
        "available_classes": list(pipeline.label_encoder.classes_) if hasattr(pipeline.label_encoder, 'classes_') else []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)