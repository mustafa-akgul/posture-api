import pandas as pd
from cnn_lstm_pipeline import CNNLSTMPipeline

pipeline = CNNLSTMPipeline(window_size=15, use_weighted_window=True)
df = pd.read_csv("Posture_API\\datasets\\new_dataset_y.csv")
pipeline.fit(df, epochs=30, batch_size=32)
pipeline.save_pipeline("pipeline")
print("✅ Model eğitildi ve kaydedildi")