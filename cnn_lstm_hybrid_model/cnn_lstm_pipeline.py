import numpy as np
from sklearn.preprocessing import LabelEncoder
from .cnn_lstm_model_setup import build_cnn_lstm_model
import os
import tensorflow as tf
import joblib

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

class CNNLSTMPipeline:
    def __init__(self, window_size=20, stride=6):
        self.window_size = window_size
        self.stride = stride
        self.model = None
        self.label_encoder = LabelEncoder()

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
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        return self

    def predict(self, df):
        X, _ = get_raw_windows(df, self.window_size, self.stride)
        y_pred = np.argmax(self.model.predict(X), axis=1)
        return self.label_encoder.inverse_transform(y_pred)

    def save_pipeline(self, filepath="pipeline.pkl"):
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen kaydetmeden önce modeli fit edin.")
        pipeline_data = {
            "model": self.model,
            "label_encoder": self.label_encoder,
            "window_size": self.window_size,
            "stride": self.stride
        }
        joblib.dump(pipeline_data, filepath)
        print(f"Pipeline '{filepath}' dosyasına kaydedildi.")

    def load_pipeline(self, filepath="pipeline.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"'{filepath}' dosyası bulunamadı.")
        pipeline_data = joblib.load(filepath)
        self.model = pipeline_data["model"]
        self.label_encoder = pipeline_data["label_encoder"]
        self.window_size = pipeline_data["window_size"]
        self.stride = pipeline_data["stride"]
        print(f"Pipeline '{filepath}' dosyasından yüklendi.")
