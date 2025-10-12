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
        
        # Adaptive smoothing için parametreler
        self.prediction_history = deque(maxlen=3)
        self.smoothing_factor = 0.7
        
        # YENİ: Değişim tespiti için
        self.last_prediction = None
        self.stable_count = 0  # Aynı tahmin kaç kez üst üste geldi
        self.transition_threshold = 0.3  # Geçiş için güven eşiği
        
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

    def detect_posture_change(self):
        """Buffer'daki veri değişimini tespit et"""
        if len(self.data_buffer) < self.window_size:
            return False
        
        buffer_array = np.array(self.data_buffer)
        
        # İlk ve son %30'luk kısımları karşılaştır
        split_point = int(self.window_size * 0.3)
        
        first_segment = buffer_array[:split_point]
        last_segment = buffer_array[-split_point:]
        
        # Ortalama ve varyans değişimi
        mean_change = np.linalg.norm(
            np.mean(last_segment, axis=0) - np.mean(first_segment, axis=0)
        )
        
        # Threshold: 0.4g'den fazla değişim (deneysel olarak ayarlanabilir)
        return mean_change > 0.4

    def predict_full(self):
        """Tam pencere ile tahmin yap - iyileştirilmiş"""
        window_data = np.array(self.data_buffer)
        window_data = window_data.reshape(1, self.window_size, 3)

        # Ham tahmin
        raw_prediction = self.model.predict(window_data, verbose=0)[0]
        
        # Değişim var mı kontrol et
        posture_change_detected = self.detect_posture_change()
        
        if posture_change_detected:
            # Değişim algılandı - smoothing'i azalt, yeni duruma hızlı geç
            effective_smoothing = 0.9  # %90 yeni veri
            # Tarihçeyi temizle
            self.prediction_history.clear()
            smoothed = raw_prediction
        else:
            # Normal smoothing
            effective_smoothing = self.smoothing_factor
            
            if len(self.prediction_history) > 0:
                # Exponential moving average
                last_pred = self.prediction_history[-1]
                smoothed = (effective_smoothing * raw_prediction + 
                           (1 - effective_smoothing) * last_pred)
            else:
                smoothed = raw_prediction
        
        # Tahmin geçmişine ekle
        self.prediction_history.append(smoothed.copy())
        
        # Final tahmin
        predicted_class = np.argmax(smoothed)
        confidence = float(np.max(smoothed))
        
        # Stabilite kontrolü
        if self.last_prediction == predicted_class:
            self.stable_count += 1
        else:
            self.stable_count = 1
        
        self.last_prediction = predicted_class
        
        # Eğer yeni tahmin çok güçlü ise (%60+ güven) hemen geç
        if confidence > 0.6 or self.stable_count >= 2:
            posture = self.label_encoder.inverse_transform([predicted_class])[0]
        else:
            # Güven düşükse bir önceki tahminde kal (flicker önleme)
            if self.last_prediction is not None:
                posture = self.label_encoder.inverse_transform([self.last_prediction])[0]
            else:
                posture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        all_predictions = {
            self.label_encoder.inverse_transform([i])[0]: round(float(smoothed[i]), 3)
            for i in range(len(smoothed))
        }
        
        return posture, confidence, all_predictions

    def add_data_point(self, x, y, z):
        """Tek veri noktası ekle - iyileştirilmiş"""
        self.data_buffer.append([x, y, z])
        current_size = len(self.data_buffer)
        
        if current_size >= self.window_size:
            return self.predict_full()   
        else:
            return None, None, None

    def predict_batch(self, df):
        """Toplu tahmin için"""
        X, _ = get_raw_windows(df, self.window_size, self.stride)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def set_smoothing_factor(self, factor: float):
        """Yumuşatma faktörünü ayarla (0.0 = çok smooth, 1.0 = çok reaktif)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))

    def set_transition_threshold(self, threshold: float):
        """Geçiş eşiğini ayarla"""
        self.transition_threshold = max(0.0, min(1.0, threshold))

    def get_prediction_stats(self) -> Dict:
        """Tahmin istatistiklerini getir - JSON safe"""
        return {
            "buffer_size": int(len(self.data_buffer)),
            "prediction_history_size": int(len(self.prediction_history)),
            "smoothing_factor": float(self.smoothing_factor),
            "window_size": int(self.window_size),
            "stable_count": int(self.stable_count),
            "last_prediction": int(self.last_prediction) if self.last_prediction is not None else None
        }

    def reset_buffer(self):
        """Buffer ve tahmin geçmişini temizle"""
        self.data_buffer.clear()
        self.prediction_history.clear()
        self.stable_count = 0
        self.last_prediction = None

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
            "smoothing_factor": self.smoothing_factor,
            "transition_threshold": self.transition_threshold
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
        self.transition_threshold = pipeline_data.get("transition_threshold", 0.3)
        
        print(f"Model '{model_path}' yüklendi.")
        print(f"Pipeline '{pickle_path}' yüklendi.")