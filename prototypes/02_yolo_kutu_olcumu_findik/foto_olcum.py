from random import random

from ultralytics import YOLO
import cv2

# Modeli yükle
model = YOLO('Nuts.pt')  # Eğitilmiş modeli yükle

# Videoyu tahmin et ve sonuçları al
sonuc = model.predict(source="4.JPG", show=True)
# Videoyu yükle
img = cv2.imread("4.JPG")
cv2.imshow("image",img)
image= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Detect results
results = model.predict([image])

#if results:
        #r= results[0]
        #object_count= len(results)
for i in results:
    x1= int(i.boxes.xyxy[0][0])
    y1 = int(i.boxes.xyxy[0][1])
    x2 = int(i.boxes.xyxy[0][2])
    y2 = int(i.boxes.xyxy[0][3])
    print(i.boxes.xyxy[0][0], "1.nut, x1")
    print(i.boxes.xyxy[0][1], "1.nut, y1")
    print(i.boxes.xyxy[0][2], "1.nut,x2")
    print(i.boxes.xyxy[0][3], "1.nut,y2")
    print(i.boxes.xyxy[1][0], "2.nut, x1")
    print(i.boxes.xyxy[1][1], "2.nut, y1")
    print(i.boxes.xyxy[1][2], "2.nut,x2")
    print(i.boxes.xyxy[1][3], "2.nut,y2")
    print(i.boxes.xyxy[2][0], "3.nut, x1")
    print(i.boxes.xyxy[2][1], "3.nut, y1")
    print(i.boxes.xyxy[2][2], "3.nut,x2")
    print(i.boxes.xyxy[2][3], "3.nut,y2")
    print(i.boxes.xyxy)
    width = x2 - x1
    ratio_px_mm = 153 / 14
    mm = width / ratio_px_mm
    cm = mm / 10
    print(cm, "cm")
#for r in results:
        #x1 = int(r.boxes.xyxy[0][0])
        #y1 = int(r.boxes.xyxy[0][1])
        #x2 = int(r.boxes.xyxy[0][2])
        #y2 = int(r.boxes.xyxy[0][3])
        #y1,x1,y2,x2= r["rois"][i]
        #cv2.rectangle(img, (x1, y1), (x2, y2), (25, 15, 220), 3)
        #Width
        #width= x2-x1
        #cv2.putText(img, str(width), (x1,y1-10), cv2.FONT_HERSHEY_PLAIN,1,(25,15,220),2)
        #1.4 CM= 153
        #14 MM =153
        #Ratio Pixels to MM
        #ratio_px_mm= 153/14
        #mm= width / ratio_px_mm
        # cm=mm/10
        #cv2.rectangle(img,(x1,y1-35),(x1+80,y1-5),(25,15,220), -1)
        #cv2.putText(img, "{} CM ".format(round(cm,2)), (x1,y1 -10), cv2.FONT_HERSHEY_PLAIN,4,(25,15,220),2)


cv2.waitKey()
cv2.destroyAllWindows()  # Pencereyi kapat
