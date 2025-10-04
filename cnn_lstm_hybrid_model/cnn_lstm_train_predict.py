import pandas as pd
from cnn_lstm_pipeline import CNNLSTMPipeline

df = pd.read_csv("cnn_lstm_hybrid_model/datasets/new_dataset.csv")

pipeline = CNNLSTMPipeline(window_size=20, stride=1)
pipeline.fit(df, epochs=30, batch_size=32)

pipeline.save_pipeline("pipeline")
print("Model eğitimi tamamlandı!")