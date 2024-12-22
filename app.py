from ultralytics import YOLO
import cv2
import time

model=YOLO("best.pt")

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise("No Camera")
while True:
    ret, image=cam.read()
    if not ret:
        break

    _time_mulai_=time.time()
    result = model.predict(image,show=True)

    print("waktu",time.time(),_time_mulai_)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
