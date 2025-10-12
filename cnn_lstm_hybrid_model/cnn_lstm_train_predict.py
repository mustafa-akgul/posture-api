import pandas as pd
from cnn_lstm_pipeline import CNNLSTMPipeline
# Veri dağılımını kontrol et:
import pandas as pd
df = pd.read_csv("Posture_API\\cnn_lstm_hybrid_model\\datasets\\new_dataset.csv")
print(df['label'].value_counts())

print("✅ Model eğitildi ve kaydedildi")