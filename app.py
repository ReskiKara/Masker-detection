import streamlit as st
from ultralytics import YOLO
import torch
import os

# Pastikan file model ada
MODEL_PATH = "best.pt"

if not os.path.isfile(MODEL_PATH):
    st.error(f"File model '{MODEL_PATH}' tidak ditemukan. Unggah file model terlebih dahulu.")
else:
    try:
        model = YOLO(MODEL_PATH)
        st.success("Model berhasil dimuat!")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")

st.title("Aplikasi Deteksi Masker dengan YOLO")
st.write("Silakan unggah gambar untuk mendeteksi masker.")
