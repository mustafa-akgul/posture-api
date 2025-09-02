import numpy as np
from sklearn.preprocessing import LabelEncoder
from cnn_lstm_model_setup import build_cnn_lstm_model
import os
import tensorflow as tf
import pickle

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
        windows, labels = get_raw_windows(df, self.window_size, self.stride)
        if fit_encoder:
            labels_encoded = self.label_encoder.fit_transform(labels)
        else:
            labels_encoded = self.label_encoder.transform(labels)
        return windows, labels_encoded

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
    
    def save_pipeline(self, filepath):
       
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi. Lütfen kaydetmeden önce modeli fit edin.")
        
        
        output_dir = os.path.dirname(filepath)
        if output_dir == '':
            output_dir = '.'  
        
        # Klasör yoksa oluştur.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Keras modelinin adını belirle
        keras_model_filename = "cnn_lstm_model.h5"
        
        # Modeli kaydet
        model_path = os.path.join(output_dir, keras_model_filename)
        self.model.save(model_path)
        
        # Pipeline'ın diğer verilerini bir sözlükte topla ve pickle ile kaydet
        pipeline_data = {
            'label_encoder': self.label_encoder,
            'window_size': self.window_size,
            'stride': self.stride,
            'keras_model_filename': keras_model_filename
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(pipeline_data, f)
        
        print(f"Pipeline meta-verisi '{filepath}' dosyasına kaydedildi.")
        print(f"Keras modeli '{model_path}' dosyasına kaydedildi.")

    def load_pipeline(self, filepath):
       
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"'{filepath}' dosyası bulunamadı.")

        # Dosya yolu boşsa, geçerli çalışma dizinini kullan.
        output_dir = os.path.dirname(filepath)
        if output_dir == '':
            output_dir = '.'
            
        # LabelEncoder ve diğer parametreleri yükle
        with open(filepath, "rb") as f:
            pipeline_data = pickle.load(f)
        
        self.label_encoder = pipeline_data['label_encoder']
        self.window_size = pipeline_data['window_size']
        self.stride = pipeline_data['stride']
        
        # Keras modelini ilişkili dosya adıyla yükle
        model_name = pipeline_data['keras_model_filename']
        self.model = tf.keras.models.load_model(os.path.join(output_dir, model_name))
        
        print(f"Pipeline '{filepath}' dosyasından yüklendi.")