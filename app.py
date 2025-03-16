import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Fungsi untuk load model dengan cache agar tidak di-load berulang
@st.cache_resource
def load_model():
    model_path = "D:\\Skripsi\\Scan-Makanan-Jepara\\jepara_food_classifier.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# Load model
model = load_model()

# Daftar label kelas (pastikan sesuai dengan train_generator.class_indices)
class_labels = ['adon_adon_coro', 'horok_horok', 'pindang_serani']

st.title("🍽️ Jepara Food Classifier")
st.write("Upload gambar makanan khas Jepara untuk diklasifikasikan.")

# Upload file gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing gambar
    img = image.resize((224, 224))  # Sesuaikan dengan ukuran input model
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.write(f"**{class_labels[predicted_class_index]}** dengan confidence **{confidence_score:.2f}**")