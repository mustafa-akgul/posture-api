import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from collections import deque
from typing import Dict, Optional


def get_raw_windows(df, window_size, stride):
    windows = []
    labels = []
    has_label = 'label' in df.columns
    
    for start in range(0, len(df) - window_size + 1, stride):
        window = df.iloc[start:start + window_size][['x', 'y', 'z']].values
        windows.append(window)
        if has_label:
            label = df.iloc[start:start + window_size]['label'].mode()[0]
            labels.append(label)
    
    if has_label:
        return np.array(windows), np.array(labels)
    else:
        return np.array(windows), None


def build_cnn_lstm_model(window_size, n_features, n_classes):
    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu', input_shape=(window_size, n_features)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class CNNLSTMPipeline:
    
    def __init__(self, window_size=15, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.model = None
        self.label_encoder = LabelEncoder()
        self.data_buffer = deque(maxlen=window_size)
        
        # Basit smoothing ayarları - hızlı tepki için
        self.prediction_history = deque(maxlen=2)  # Sadece son 2 tahmin
        self.smoothing_factor = 0.7  # %70 yeni, %30 eski

    def prepare_data(self, df, fit_encoder=False):
        X, y = get_raw_windows(df, self.window_size, self.stride)
        
        if y is not None:
            if fit_encoder:
                y_encoded = self.label_encoder.fit_transform(y)
            else:
                y_encoded = self.label_encoder.transform(y)
            return X, y_encoded
        else:
            return X, None

    def fit(self, df, epochs=20, batch_size=32):
        X, y = self.prepare_data(df, fit_encoder=True)
        n_classes = len(np.unique(y))
        
        self.model = build_cnn_lstm_model(self.window_size, X.shape[2], n_classes)
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        return history

    def predict_partial(self):
        """Kısmi pencere ile tahmin yap (10+ veri varsa)"""
        current_size = len(self.data_buffer)
        if current_size < 10:
            return None, None, None
            
        # Eksik kısmı son veriyle doldur
        temp_buffer = list(self.data_buffer)
        while len(temp_buffer) < self.window_size:
            temp_buffer.append(temp_buffer[-1])
        
        window_data = np.array(temp_buffer)
        window_data = window_data.reshape(1, self.window_size, 3)
        
        raw_prediction = self.model.predict(window_data, verbose=0)
        predicted_class = np.argmax(raw_prediction[0])
        confidence = float(raw_prediction[0][predicted_class]) * 0.6  # Kısmi güven
        
        posture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        all_predictions = {
            self.label_encoder.inverse_transform([i])[0]: round(float(raw_prediction[0][i]), 3)
            for i in range(len(raw_prediction[0]))
        }
        
        return posture, confidence, all_predictions

    def predict_full(self):
        """Tam pencere ile tahmin yap"""
        window_data = np.array(self.data_buffer)
        window_data = window_data.reshape(1, self.window_size, 3)

        # Ham tahmin
        raw_prediction = self.model.predict(window_data, verbose=0)
        
        # Basit smoothing
        if len(self.prediction_history) > 0:
            last_pred = self.prediction_history[-1]
            smoothed = (self.smoothing_factor * raw_prediction[0] + 
                       (1 - self.smoothing_factor) * last_pred)
        else:
            smoothed = raw_prediction[0]
        
        # Tahmin geçmişine ekle
        self.prediction_history.append(raw_prediction[0])
        
        # Final tahmin
        predicted_class = np.argmax(smoothed)
        confidence = float(np.max(smoothed))
        
        posture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        all_predictions = {
            self.label_encoder.inverse_transform([i])[0]: round(float(smoothed[i]), 3)
            for i in range(len(smoothed))
        }
        
        return posture, confidence, all_predictions

    def add_data_point(self, x, y, z):
        """Tek veri noktası ekle"""
        self.data_buffer.append([x, y, z])
        current_size = len(self.data_buffer)
        
        if current_size >= self.window_size:
            return self.predict_full()
        elif current_size >= 10:
            return self.predict_partial()
        else:
            return None, None, None

    def predict_batch(self, df):
        """Toplu tahmin için"""
        X, _ = get_raw_windows(df, self.window_size, self.stride)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def set_smoothing_factor(self, factor: float):
        """Yumuşatma faktörünü ayarla"""
        self.smoothing_factor = max(0.0, min(1.0, factor))

    def get_prediction_stats(self) -> Dict:
        """Tahmin istatistiklerini getir"""
        return {
            "buffer_size": len(self.data_buffer),
            "prediction_history_size": len(self.prediction_history),
            "smoothing_factor": self.smoothing_factor,
            "window_size": self.window_size
        }

    def reset_buffer(self):
        """Buffer ve tahmin geçmişini temizle"""
        self.data_buffer.clear()
        self.prediction_history.clear()

    def save_pipeline(self, filepath="pipeline"):
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi.")
        
        model_path = f"{filepath}_model.keras"
        self.model.save(model_path)
        
        pipeline_data = {
            "label_encoder": self.label_encoder,
            "window_size": self.window_size,
            "stride": self.stride,
            "model_path": model_path,
            "smoothing_factor": self.smoothing_factor
        }
        
        pickle_path = f"{filepath}.pkl"
        joblib.dump(pipeline_data, pickle_path)
        
        print(f"Model '{model_path}' kaydedildi.")
        print(f"Pipeline '{pickle_path}' kaydedildi.")

    def load_pipeline(self, filepath="pipeline"):
        pickle_path = f"{filepath}.pkl"
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"'{pickle_path}' bulunamadı.")
        
        pipeline_data = joblib.load(pickle_path)
        
        model_path = pipeline_data["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası '{model_path}' bulunamadı.")
        
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = pipeline_data["label_encoder"]
        self.window_size = pipeline_data["window_size"]
        self.stride = pipeline_data["stride"]
        self.smoothing_factor = pipeline_data.get("smoothing_factor", 0.7)
        
        print(f"Model '{model_path}' yüklendi.")
        print(f"Pipeline '{pickle_path}' yüklendi.")