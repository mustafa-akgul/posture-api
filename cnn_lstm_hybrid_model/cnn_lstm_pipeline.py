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
    
    def __init__(self, window_size=20, stride=1):
        self.window_size = window_size
        self.stride = stride
        self.model = None
        self.label_encoder = LabelEncoder()
        self.data_buffer = deque(maxlen=window_size)
        
        # Ağırlıklandırma ayarları
        self.temporal_weights = np.linspace(0.7, 1.3, window_size).reshape(-1, 1)
        self.prediction_history = deque(maxlen=5)  # Son 5 tahmin
        self.smoothing_factor = 0.7

    def apply_temporal_weighting(self, window_data: np.ndarray) -> np.ndarray:
        """
        Zamansal ağırlıklandırma uygula
        Son verilere daha fazla önem verir
        """
        weighted_data = window_data * self.temporal_weights
        return weighted_data

    def smooth_predictions(self, current_prediction: np.ndarray) -> np.ndarray:
        """
        Tahminleri yumuşatmak için moving average uygula
        """
        if len(self.prediction_history) == 0:
            return current_prediction
        
        # Önceki tahminlerin ortalamasını al
        historical_avg = np.mean(np.array(list(self.prediction_history)), axis=0)
        
        # Mevcut tahmin ve geçmiş ortalamayı birleştir
        smoothed = (self.smoothing_factor * current_prediction + 
                    (1 - self.smoothing_factor) * historical_avg)
        
        return smoothed

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

    def add_data_point(self, x, y, z):
        """Tek veri noktası ekle ve tahmin yap"""
        self.data_buffer.append([x, y, z])
        
        if len(self.data_buffer) == self.window_size:
            posture, confidence, all_predictions = self.predict_current()
            return posture, confidence, all_predictions
        else:
            return None, None, None
        
    def predict_current(self):
        """Mevcut buffer için ağırlıklı tahmin yap"""
        if len(self.data_buffer) < self.window_size:
            raise ValueError(f"Buffer'da yeterli veri yok. Gerekli: {self.window_size}, Mevcut: {len(self.data_buffer)}")
        
        # 1. Ham veriyi al
        window_data = np.array(self.data_buffer)
        
        # 2. Zamansal ağırlıklandırma uygula
        weighted_data = self.apply_temporal_weighting(window_data)
        weighted_data = weighted_data.reshape(1, self.window_size, 3)

        # 3. Ham tahmin
        raw_prediction = self.model.predict(weighted_data, verbose=0)
        
        # 4. Tahmin yumuşatma uygula
        smoothed_prediction = self.smooth_predictions(raw_prediction)
        
        # 5. Tahmin geçmişine ekle
        self.prediction_history.append(raw_prediction[0])
        
        # 6. Final tahmin
        final_prediction = smoothed_prediction
        predicted_class = np.argmax(final_prediction[0])
        confidence = float(np.max(final_prediction[0]))
        
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        # 7. Tüm sınıfların detaylı tahminlerini hazırla
        all_predictions = {
            self.label_encoder.inverse_transform([i])[0]: {
                "raw_prob": round(float(raw_prediction[0][i]), 3),
                "final_prob": round(float(final_prediction[0][i]), 3)
            }
            for i in range(len(raw_prediction[0]))
        }
        
        return predicted_label, confidence, all_predictions

    def predict_batch(self, df):
        """Toplu tahmin için (ağırlıklandırma olmadan)"""
        X, _ = get_raw_windows(df, self.window_size, self.stride)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def set_smoothing_factor(self, factor: float):
        """Yumuşatma faktörünü ayarla (0.0 - 1.0)"""
        self.smoothing_factor = max(0.0, min(1.0, factor))

    def set_temporal_weights(self, weights: np.ndarray):
        """Zamansal ağırlıkları ayarla"""
        if len(weights) != self.window_size:
            raise ValueError(f"Ağırlık boyutu pencere boyutuyla eşleşmeli: {self.window_size}")
        self.temporal_weights = np.array(weights).reshape(-1, 1)

    def get_prediction_stats(self) -> Dict:
        """Tahmin istatistiklerini getir"""
        return {
            "buffer_size": len(self.data_buffer),
            "prediction_history_size": len(self.prediction_history),
            "smoothing_factor": self.smoothing_factor,
            "temporal_weights": self.temporal_weights.flatten().tolist()
        }

    def reset_buffer(self):
        """Buffer ve tahmin geçmişini temizle"""
        self.data_buffer.clear()
        self.prediction_history.clear()

    def save_pipeline(self, filepath="pipeline"):
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen kaydetmeden önce modeli fit edin.")
        
        model_path = f"{filepath}_model.keras"
        self.model.save(model_path)
        
        pipeline_data = {
            "label_encoder": self.label_encoder,
            "window_size": self.window_size,
            "stride": self.stride,
            "model_path": model_path,
            "temporal_weights": self.temporal_weights,
            "smoothing_factor": self.smoothing_factor
        }
        
        pickle_path = f"{filepath}.pkl"
        joblib.dump(pipeline_data, pickle_path)
        
        print(f"Model '{model_path}' dosyasına kaydedildi.")
        print(f"Pipeline '{pickle_path}' dosyasına kaydedildi.")

    def load_pipeline(self, filepath="pipeline"):
        pickle_path = f"{filepath}.pkl"
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"'{pickle_path}' dosyası bulunamadı.")
        
        pipeline_data = joblib.load(pickle_path)
        
        model_path = pipeline_data["model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası '{model_path}' bulunamadı.")
        
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = pipeline_data["label_encoder"]
        self.window_size = pipeline_data["window_size"]
        self.stride = pipeline_data["stride"]
        self.temporal_weights = pipeline_data.get("temporal_weights", 
                                                 np.linspace(0.7, 1.3, self.window_size).reshape(-1, 1))
        self.smoothing_factor = pipeline_data.get("smoothing_factor", 0.7)
        
        print(f"Model '{model_path}' dosyasından yüklendi.")
        print(f"Pipeline '{pickle_path}' dosyasından yüklendi.")