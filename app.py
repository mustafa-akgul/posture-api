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
pipeline = CNNLSTMPipeline()
try:
    pipeline.load_pipeline("pipeline")
    print("✅ Model yüklendi")
except Exception as e:
    print(f"⚠️ Model yüklenemedi: {e}")
    df = pd.read_csv("cnn_lstm_hybrid_model/datasets/new_dataset.csv")
    pipeline.fit(df, epochs=30, batch_size=32)
    pipeline.save_pipeline("pipeline")
    print("✅ Model eğitildi")

# Sliding window buffer
WINDOW_SIZE = 20
data_buffer = deque(maxlen=WINDOW_SIZE)

@app.get("/")
def root():
    return {
        "status": "running",
        "buffer_size": len(data_buffer),
        "window_size": WINDOW_SIZE
    }

@app.post("/predict")
def predict(data: SensorData):
    """Gerçek zamanlı tahmin - Sliding window"""
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
        window_data = np.array(data_buffer)
        scaled_data = pipeline.scaler.transform(window_data)
        X = scaled_data.reshape(1, WINDOW_SIZE, 3)
        
        prediction = pipeline.model.predict(X, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        
        posture = pipeline.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            "status": "ok",
            "posture": posture,
            "confidence": round(confidence, 3),
            "buffer_size": current_size
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_buffer():
    """Buffer'ı sıfırla"""
    data_buffer.clear()
    return {
        "status": "success",
        "message": "Buffer sıfırlandı",
        "buffer_size": 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)