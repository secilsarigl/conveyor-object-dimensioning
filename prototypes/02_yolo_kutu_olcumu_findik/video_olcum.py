from random import random

from ultralytics import YOLO
import cv2
# Modeli yükle
model = YOLO('Nuts.pt')  # Eğitilmiş modeli yükle
# Videoyu tahmin et ve sonuçları al
sonuc = model.predict(source="Video2.mp4",stream=True, show=True)
# Videoyu yükle
cap = cv2.VideoCapture("Video2.mp4")

while True:
    ret,img = cap.read()
    if not ret:
        break
    image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Detect results
    results = model.predict([image])

    if results:
       r= results[0]
       object_count= len(r["class_ids"])

    for i in range(object_count):
        y1,x1,y2,x2= r["rois"][i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (25, 15, 220), 3)
        #Width
        width= x2-x1
        #cv2.putText(img, str(width), (x1,y1-10), cv2.FONT_HERSHEY_PLAIN,1,(25,15,220),2)
        #1.4 CM= 153
        #14 MM =153
        #Ratio Pixels to MM
        ratio_px_mm= 153/14
        mm= width / ratio_px_mm
        cm=mm/10
        cv2.rectangle(img,(x1,y1-35),(x1+80,y1-5),(25,15,220), -1)
        cv2.putText(img, "{} CM ".format(round(cm,2)), (x1,y1 -10), cv2.FONT_HERSHEY_PLAIN,4,(25,15,220),2)

cap.release()
cv2.destroyAllWindows()  # Pencereyi kapat
