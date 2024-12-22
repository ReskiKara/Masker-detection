import streamlit as st
from ultralytics import YOLO
import cv2
import time

# Load Model
model = YOLO("best.pt")

# Streamlit UI
st.title("YOLO Object Detection")

# Access Webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    st.error("Camera not accessible")
    st.stop()

stframe = st.empty()
while True:
    ret, image = cam.read()
    if not ret:
        st.error("Failed to read from camera")
        break

    result = model.predict(image, show=False)
    stframe.image(image, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
