import streamlit as st
from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("best.pt")

# Streamlit UI
st.title("YOLO Object Detection with Streamlit")

# Access Webcam
stframe = st.empty()
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    st.error("Camera not accessible!")
    st.stop()

while True:
    ret, image = cam.read()
    if not ret:
        st.error("Failed to read from camera")
        break

    # Predict with YOLO
    results = model.predict(image, show=False)

    # Display Image
    stframe.image(image, channels="BGR")

    # Exit Loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
