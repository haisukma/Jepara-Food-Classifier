# import tensorflow as tf

# model_path = "D:\\Skripsi\\Scan-Makanan-Jepara\\jepara_food_classifier.h5"
# model = tf.keras.models.load_model(model_path)

# print("✅ Model berhasil dimuat!")

import tensorflow as tf

model_path = "D:\\Skripsi\\Scan-Makanan-Jepara\\jepara_food_classifier.h5"
model = tf.keras.models.load_model(model_path, compile=False)

print("✅ Model berhasil dimuat!")
