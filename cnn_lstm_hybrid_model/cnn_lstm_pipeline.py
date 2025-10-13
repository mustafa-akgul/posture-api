import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from collections import deque
from typing import Dict, Optional
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping


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
        layers.Dropout(0.4),
        layers.Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class CNNLSTMPipeline:
    
    def __init__(self, window_size=12, stride=1, use_weighted_window=True, 
                 normalize=True, auto_clean=True):
        self.window_size = window_size
        self.stride = stride
        self.model = None
        self.label_encoder = LabelEncoder()
        self.data_buffer = deque(maxlen=window_size)
        self.normalize = normalize
        self.auto_clean = auto_clean
        self.scaler = StandardScaler() if normalize else None
        # Weighted window için ağırlıklar
        self.use_weighted_window = use_weighted_window
        if use_weighted_window:
            self.window_weights = np.exp(np.linspace(-2, 0, window_size))
            self.window_weights = self.window_weights / self.window_weights.sum()
        else:
            self.window_weights = np.ones(window_size) / window_size
        
        # Adaptive smoothing
        self.prediction_history = deque(maxlen=3)
        self.smoothing_factor = 0.7
        
        # Değişim tespiti
        self.last_prediction = None
        self.stable_count = 0
        self.change_detected_count = 0
        
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

    def fit(self, df, epochs=30, batch_size=32, 
            use_class_weights=True,
            use_augmentation=True,
            use_smote=False):
        
        # 1. Veri temizleme
        if self.auto_clean:
            df = self.clean_outliers(df)
            df = self.remove_sudden_jumps(df)
        
        # 2. Normalizasyon
        if self.normalize:
            self.scaler.fit(df[['x', 'y', 'z']])
            df[['x', 'y', 'z']] = self.scaler.transform(df[['x', 'y', 'z']])
        
        # 3. Pencere oluşturma
        X, y = self.prepare_data(df, fit_encoder=True)
        
        # 4. Noise augmentation
        if use_augmentation:
            X, y = self.augment_data(X, y)
        
        # 5. SMOTE
        if use_smote:
            X, y = self.apply_smote(X, y)
        
        # 6. Class weights
        class_weights = None
        if use_class_weights:
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights = compute_class_weight('balanced', classes=classes, y=y)
            class_weights = dict(enumerate(weights))
        
        # 7. Model eğitimi
        self.model = build_cnn_lstm_model(self.window_size, X.shape[2], len(np.unique(y)))
        
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # 8. Validation metrikleri
        self._print_validation_metrics(X, y)
        
        return history

    def detect_strong_change(self):
        """Güçlü duruş değişimini tespit et"""
        if len(self.data_buffer) < self.window_size:
            return False
        
        buffer_array = np.array(self.data_buffer)
        
        # Son %40 vs İlk %40
        segment_size = int(self.window_size * 0.4)
        
        first_segment = buffer_array[:segment_size]
        last_segment = buffer_array[-segment_size:]
        
        # Ortalama değişim
        mean_change = np.linalg.norm(
            np.mean(last_segment, axis=0) - np.mean(first_segment, axis=0)
        )
        
        # Varyans değişimi
        var_change = np.linalg.norm(
            np.var(last_segment, axis=0) - np.var(first_segment, axis=0)
        )
        
        # ✅ FİX: Threshold düşürüldü
        change_detected = (mean_change > 0.25) or (var_change > 0.12)  # 0.3→0.25, 0.15→0.12
        
        if change_detected:
            self.change_detected_count += 1
        else:
            self.change_detected_count = max(0, self.change_detected_count - 1)
        
        # ✅ FİX: 1 kez yeterli (2 yerine)
        return self.change_detected_count >= 1

    def predict_with_weighted_window(self, window_data):
        """Ağırlıklı pencere ile tahmin"""
        if not self.use_weighted_window:
            return self.model.predict(window_data, verbose=0)[0]
        
        predictions = []
        weights = []
        
        # ✅ FİX: Ağırlıklar değişti [0.3→0.2, 0.5→0.5, 0.2→0.3]
        
        # 1. Tam pencere (ağırlık: 0.2) - AZALTILDI
        pred_full = self.model.predict(window_data, verbose=0)[0]
        predictions.append(pred_full)
        weights.append(0.2)
        
        # 2. Son 10 veri (ağırlık: 0.5) - AYNI
        if len(self.data_buffer) >= 10:
            recent_window = np.array(list(self.data_buffer)[-10:])
            padded = np.pad(recent_window, ((self.window_size - 10, 0), (0, 0)), 
                        mode='edge')
            pred_recent = self.model.predict(padded.reshape(1, self.window_size, 3), 
                                            verbose=0)[0]
            predictions.append(pred_recent)
            weights.append(0.5)
        
        # 3. Son 5 veri (ağırlık: 0.3) - ARTIRILDI
        if len(self.data_buffer) >= 5:
            latest_window = np.array(list(self.data_buffer)[-5:])
            padded = np.pad(latest_window, ((self.window_size - 5, 0), (0, 0)), 
                        mode='edge')
            pred_latest = self.model.predict(padded.reshape(1, self.window_size, 3), 
                                            verbose=0)[0]
            predictions.append(pred_latest)
            weights.append(0.3)
        
        # Ağırlıklı ortalama
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_pred = np.zeros_like(predictions[0])
        for pred, w in zip(predictions, weights):
            weighted_pred += pred * w
        
        return weighted_pred

    def predict_full(self):
        """Tam pencere ile tahmin yap"""
        window_data = np.array(self.data_buffer)
        window_data = window_data.reshape(1, self.window_size, 3)

        raw_prediction = self.predict_with_weighted_window(window_data)
        strong_change = self.detect_strong_change()
        
        if strong_change:
            effective_smoothing = 0.95
            self.prediction_history.clear()
            smoothed = raw_prediction
            print(f"🔥 GÜÇLÜ DEĞİŞİM TESPİT EDİLDİ!")
        else:
            effective_smoothing = self.smoothing_factor
            
            if len(self.prediction_history) > 0:
                last_pred = self.prediction_history[-1]
                smoothed = (effective_smoothing * raw_prediction + 
                        (1 - effective_smoothing) * last_pred)
            else:
                smoothed = raw_prediction
        
        self.prediction_history.append(smoothed.copy())
        
        predicted_class = np.argmax(smoothed)
        confidence = float(np.max(smoothed))
        
        # ✅ YENİ: Hangi sınıftan hangi sınıfa geçiş yapılıyor kontrol et
        current_posture_name = self.label_encoder.inverse_transform([predicted_class])[0] if predicted_class is not None else None
        last_posture_name = self.label_encoder.inverse_transform([self.last_prediction])[0] if self.last_prediction is not None else None
        
        # ✅ YENİ: Kritik geçişler için threshold'u düşür
        is_critical_transition = False
        if last_posture_name and current_posture_name:
            critical_transitions = [
                ("normal", "slouch"),
                ("normal", "hunch"),
                ("slouch", "hunch"),
                ("hunch", "slouch")
            ]
            if (last_posture_name, current_posture_name) in critical_transitions:
                is_critical_transition = True
                print(f"⚠️ Kritik geçiş: {last_posture_name} → {current_posture_name}")
        
        # Threshold'u dinamik ayarla
        if is_critical_transition or strong_change:
            required_confidence = 0.45  # Düşük threshold
        else:
            required_confidence = 0.55  # Normal threshold
        
        # Stabilite kontrolü
        is_different_prediction = (self.last_prediction is not None and 
                                self.last_prediction != predicted_class)
        
        if is_different_prediction:
            if confidence > required_confidence or strong_change:
                self.stable_count = 1
                self.last_prediction = predicted_class
                print(f"⚡ Geçiş: {last_posture_name} → {current_posture_name} (güven: {confidence:.2f})")
            else:
                self.stable_count = 0
        else:
            if self.last_prediction == predicted_class:
                self.stable_count += 1
            else:
                self.last_prediction = predicted_class
                self.stable_count = 1
        
        # Final posture
        if self.stable_count >= 1 and confidence > required_confidence:
            final_posture = self.label_encoder.inverse_transform([predicted_class])[0]
        elif self.last_prediction is not None and not is_different_prediction:
            final_posture = self.label_encoder.inverse_transform([predicted_class])[0]
        elif self.last_prediction is not None:
            final_posture = self.label_encoder.inverse_transform([self.last_prediction])[0]
        else:
            final_posture = self.label_encoder.inverse_transform([predicted_class])[0]
        
        all_predictions = {
            self.label_encoder.inverse_transform([i])[0]: round(float(smoothed[i]), 3)
            for i in range(len(smoothed))
        }
        
        return final_posture, confidence, all_predictions

    def add_data_point(self, x, y, z):
        """Tek veri noktası ekle"""
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
        """Yumuşatma faktörünü ayarla"""
        self.smoothing_factor = max(0.0, min(1.0, factor))

    def set_weighted_window(self, enabled: bool):
        """Weighted window'u aç/kapat"""
        self.use_weighted_window = enabled

    def get_prediction_stats(self) -> Dict:
        """Tahmin istatistiklerini getir"""
        return {
            "buffer_size": int(len(self.data_buffer)),
            "prediction_history_size": int(len(self.prediction_history)),
            "smoothing_factor": float(self.smoothing_factor),
            "window_size": int(self.window_size),
            "stable_count": int(self.stable_count),
            "last_prediction": int(self.last_prediction) if self.last_prediction is not None else None,
            "change_detected_count": int(self.change_detected_count),
            "use_weighted_window": bool(self.use_weighted_window)
        }

    def reset_buffer(self, hard_reset=False):
        """Buffer ve tahmin geçmişini temizle"""
        self.data_buffer.clear()
        self.prediction_history.clear()
        self.stable_count = 0
        self.last_prediction = None
        self.change_detected_count = 0
        
        if hard_reset:
            if self.model is not None:
                self.model.reset_states() if hasattr(self.model, 'reset_states') else None
                print("🔄 Model internal state temizlendi")

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
            "use_weighted_window": self.use_weighted_window
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
        self.use_weighted_window = pipeline_data.get("use_weighted_window", True)
        
        if self.use_weighted_window:
            self.window_weights = np.exp(np.linspace(-2, 0, self.window_size))
            self.window_weights = self.window_weights / self.window_weights.sum()
        
        print(f"Model '{model_path}' yüklendi.")
        print(f"Pipeline '{pickle_path}' yüklendi.")
        print(f"Weighted Window: {self.use_weighted_window}")