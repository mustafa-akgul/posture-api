import pandas as pd
from cnn_lstm_pipeline import CNNLSTMPipeline

df = pd.read_csv("datasets/new_dataset_x.csv")

pipeline = CNNLSTMPipeline(window_size=20, stride=6)
pipeline.fit(df, epochs=20, batch_size=32)


test_df = pd.read_csv("datasets/test_x.csv")

predictions = pipeline.predict(test_df)
print(predictions)

data = 'e'
model = 'cnn+lstm'

predictions_df = pd.DataFrame(predictions, columns=['predicted_label'])
predictions_df.to_csv(f"predictions/{data}_{model}_predictions.csv", index=False)
print(f"\nTahminler 'predictions/{data}_{model}_predictions.csv' dosyasına kaydedildi.")