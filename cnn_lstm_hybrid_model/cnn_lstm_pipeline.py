import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
from collections import deque


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
        # CNN 
        layers.Conv1D(32, 3, activation='relu', input_shape=(window_size, n_features)),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        
        # LSTM 
        layers.LSTM(64, return_sequences=False),
        
        # Dense 
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

        self.data_buffer.append([x, y, z])
        
        if len(self.data_buffer) == self.window_size:
            posture, confidence = self.predict_current()
            return posture, confidence
        else:
            return None, None
        
    def predict_current(self):
        if len(self.data_buffer) < self.window_size:
            raise ValueError(f"Buffer'da yeterli veri yok. Gerekli: {self.window_size}, Mevcut: {len(self.data_buffer)}")
        
        window_data = np.array(self.data_buffer)
        window_data = window_data.reshape(1, self.window_size, 3)

        prediction = self.model.predict(window_data, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction, axis=1)[0])

        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_label, confidence

    def predict_batch(self, df):
        X, _ = get_raw_windows(df, self.window_size, self.stride)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def save_pipeline(self, filepath="pipeline"):
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen kaydetmeden önce modeli fit edin.")
        
        model_path = f"{filepath}_model.keras"
        self.model.save(model_path)
        
        pipeline_data = {
            "label_encoder": self.label_encoder,
            "window_size": self.window_size,
            "stride": self.stride,
            "model_path": model_path
        }
        
        pickle_path = f"{filepath}.pkl"
        joblib.dump(pipeline_data, pickle_path)
        
        print(f"Model '{model_path}' dosyasına kaydedildi.")
        print(f"Pipeline '{pickle_path}' dosyasına kaydedildi.")

    def load_pipeline(self, filepath="pipeline"):
        """Pipeline'ı ve modeli yükler."""
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
        
        print(f"Model '{model_path}' dosyasından yüklendi.")
        print(f"Pipeline '{pickle_path}' dosyasından yüklendi.")

    def reset_buffer(self):
        """Veri buffer'ını temizler."""
        self.data_buffer.clear()
        print("Buffer temizlendi.")