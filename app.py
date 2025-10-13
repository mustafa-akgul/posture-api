from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional


class SensorData(BaseModel):
    x: float
    y: float
    z: float

class SmoothingConfig(BaseModel):
    smoothing_factor: float


def convert_numpy_types(obj):
    """NumPy tiplerini JSON-safe tiplere dönüştür"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


app = FastAPI(title="Real-time Posture API v2.1")

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

from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline

pipeline = CNNLSTMPipeline(window_size=12, use_weighted_window=True)

try : 
    pipeline.load_pipeline("pipeline")
    print("✅ Model yüklendi")
except Exception as e:
    if not os.path.exists("cnn_lstm_hybrid_model/datasets/new_dataset.csv"):
        raise RuntimeError("Dataset dosyası bulunamadı. Model eğitilemedi.")
    print(f"⚠️ Model yüklenemedi: {e}")
    print("Yeniden eğitiliyor...")
    df = pd.read_csv("cnn_lstm_hybrid_model/datasets/new_dataset_y.csv")
    pipeline.fit(df, epochs=30, batch_size=32)
    pipeline.save_pipeline("pipeline")
    print("✅ Model eğitildi ve kaydedildi")



@app.get("/")
def root():
    """API durumu"""
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    return {
        "status": "running",
        "version": "2.1-weighted-window",
        "features": [
            "weighted_sub_window_prediction",
            "strong_change_detection",
            "adaptive_smoothing",
            "stability_tracking"
        ],
        "pipeline_stats": stats
    }


@app.post("/predict")

def predict(data: Optional[SensorData] = None):
    try:
        posture, confidence, all_predictions = None, None, None # Değişkenleri başlangıçta tanımla

        if data is not None:
            posture, confidence, all_predictions = pipeline.add_data_point(
                data.x, data.y, data.z
            )

        # EĞER TAHMİN YAPILACAK KADAR VERİ YOKSA, "VERİ TOPLANIYOR" YANITI DÖN
        if posture is None:
            stats = pipeline.get_prediction_stats()
            return {
                "status": "collecting",
                "message": "Kalibrasyon için veri toplanıyor...",
                "buffer_size": stats.get("buffer_size", 0),
                "window_size": stats.get("window_size", 12),
                "progress_percentage": round((stats.get("buffer_size", 0) / stats.get("window_size", 12)) * 100, 1)
            }

        # Eğer veri varsa, normal yanıtı oluşturmaya devam et
        stats = pipeline.get_prediction_stats()
        stats = convert_numpy_types(stats)

        return {
            "status": "ok",
            "posture": posture,
            "confidence": round(float(confidence), 3),
            "buffer_size": len(pipeline.data_buffer),
            "all_predictions": all_predictions,
            "stability": {
                "stable_count": stats["stable_count"],
                "is_stable": stats["stable_count"] >= 2,
                "last_prediction": stats["last_prediction"]
            },
            # ... (diğer bilgiler)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update_smoothing")
def update_smoothing(config: SmoothingConfig):
    """Yumuşatma faktörünü güncelle"""
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


@app.get("/pipeline_stats")
def get_pipeline_stats():
    """Pipeline istatistikleri"""
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    buffer_percentage = (stats["buffer_size"] / stats["window_size"]) * 100
    stability_status = "stable" if stats["stable_count"] >= 2 else "transitioning"
    
    return {
        **stats,
        "buffer_percentage": round(buffer_percentage, 1),
        "stability_status": stability_status,
        "is_ready": stats["buffer_size"] >= stats["window_size"]
    }


@app.post("/reset")
def reset_buffer():
    """Soft Reset"""
    pipeline.reset_buffer(hard_reset=False)
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    return {
        "status": "success",
        "reset_type": "soft",
        "message": "Buffer temizlendi",
        "current_stats": stats
    }


@app.post("/reset/hard")
def hard_reset_buffer():
    """Hard Reset (Önerilen)"""
    pipeline.reset_buffer(hard_reset=True)
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    return {
        "status": "success",
        "reset_type": "hard",
        "message": "Pipeline tamamen sıfırlandı",
        "current_stats": stats
    }


@app.post("/reset/full")
def full_reset():
    """Full Reset - Pipeline'ı yeniden yükle"""
    global pipeline
    
    try:
        del pipeline
        pipeline = CNNLSTMPipeline(window_size=12, use_weighted_window=True)
        pipeline.load_pipeline("pipeline")
        
        stats = pipeline.get_prediction_stats()
        stats = convert_numpy_types(stats)
        
        return {
            "status": "success",
            "reset_type": "full",
            "message": "Pipeline yeniden yüklendi",
            "current_stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Full reset hatası: {str(e)}")


@app.get("/debug/buffer")
def debug_buffer():
    """Buffer içeriğini göster"""
    buffer_data = list(pipeline.data_buffer)
    
    return {
        "buffer_size": len(buffer_data),
        "max_buffer_size": pipeline.window_size,
        "buffer_content": [
            {
                "index": i,
                "x": round(float(d[0]), 4), 
                "y": round(float(d[1]), 4), 
                "z": round(float(d[2]), 4)
            } 
            for i, d in enumerate(buffer_data)
        ],
        "stable_count": int(pipeline.stable_count),
        "last_prediction": int(pipeline.last_prediction) if pipeline.last_prediction is not None else None,
        "change_detected_count": int(pipeline.change_detected_count)
    }


@app.get("/debug/last_predictions")
def debug_last_predictions():
    """Son tahminleri göster"""
    history = list(pipeline.prediction_history)
    
    formatted_history = []
    for pred_vector in history:
        pred_dict = {
            pipeline.label_encoder.inverse_transform([i])[0]: round(float(pred_vector[i]), 3)
            for i in range(len(pred_vector))
        }
        formatted_history.append(pred_dict)
    
    return {
        "prediction_history_size": len(formatted_history),
        "smoothing_factor": float(pipeline.smoothing_factor),
        "history": formatted_history
    }


@app.get("/health")
def health_check():
    """Sistem sağlık kontrolü"""
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    return {
        "status": "healthy" if pipeline.model is not None else "unhealthy",
        "model_loaded": pipeline.model is not None,
        "buffer_ready": stats["buffer_size"] >= stats["window_size"],
        "timestamp": pd.Timestamp.now().isoformat()
    }


@app.get("/config")
def get_config():
    """Konfigürasyon"""
    stats = pipeline.get_prediction_stats()
    stats = convert_numpy_types(stats)
    
    return {
        "window_size": stats["window_size"],
        "smoothing_factor": stats["smoothing_factor"],
        "use_weighted_window": stats.get("use_weighted_window", True)
    }


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
