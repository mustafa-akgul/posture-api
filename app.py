from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from collections import deque
from cnn_lstm_hybrid_model.cnn_lstm_pipeline import CNNLSTMPipeline
import uvicorn

class SensorData(BaseModel):
    x: float
    y: float
    z: float

app = FastAPI(title="Real-time Posture API")

# Model yükleme
pipeline = CNNLSTMPipeline(window_size=20)
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

# Sliding window buffer (API seviyesinde)
WINDOW_SIZE = 20
data_buffer = deque(maxlen=WINDOW_SIZE)

@app.get("/")
def root():
    return {
        "status": "running",
        "buffer_size": len(data_buffer),
        "window_size": WINDOW_SIZE,
        "pipeline_buffer": len(pipeline.data_buffer)
    }

@app.post("/predict")
def predict(data: SensorData):
    """
    Gerçek zamanlı tahmin - Sliding window
    Pipeline'ın kendi buffer'ını kullanmayıp API seviyesinde yönetiyoruz
    """
    try:
        # Veriyi buffer'a ekle (otomatik olarak en eski veri silinir)
        data_buffer.append([data.x, data.y, data.z])
        
        current_size = len(data_buffer)
        
        # İlk 20 veri toplanana kadar bekleme
        if current_size < WINDOW_SIZE:
            return {
                "status": "collecting",
                "message": f"Kalibrasyon ({current_size}/{WINDOW_SIZE})",
                "buffer_size": current_size
            }
        
        # Buffer doldu - Tahmin yap
        # Buffer'ı numpy array'e çevir
        window_data = np.array(data_buffer)  # Shape: (20, 3)
        
        # Pipeline'ın model'ini direkt kullan
        window_data_reshaped = window_data.reshape(1, WINDOW_SIZE, 3)
        
        # Tahmin
        prediction = pipeline.model.predict(window_data_reshaped, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        # Label'a çevir
        posture = pipeline.label_encoder.inverse_transform([predicted_class])[0]
        
        # Debug: Tüm sınıfların olasılıklarını göster
        all_predictions = {
            pipeline.label_encoder.inverse_transform([i])[0]: round(float(prediction[0][i]), 3)
            for i in range(len(prediction[0]))
        }
        
        print(f"📊 Tahmin Dağılımı: {all_predictions}")
        print(f"📍 Buffer son 3 veri: {list(data_buffer)[-3:]}")
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": round(confidence, 3),
            "buffer_size": current_size,
            "all_predictions": all_predictions  # Debug için
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_buffer():
    """Buffer'ı sıfırla"""
    data_buffer.clear()
    pipeline.reset_buffer()  # Pipeline'ın kendi buffer'ını da temizle
    return {
        "status": "success",
        "message": "Buffer sıfırlandı",
        "buffer_size": 0
    }

@app.get("/status")
def get_status():
    """Sistem durumu"""
    return {
        "api_buffer_size": len(data_buffer),
        "pipeline_buffer_size": len(pipeline.data_buffer),
        "window_size": WINDOW_SIZE,
        "model_loaded": pipeline.model is not None,
        "classes": list(pipeline.label_encoder.classes_) if hasattr(pipeline.label_encoder, 'classes_') else []
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)