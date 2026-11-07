from ultralytics import YOLO
import cv2
import numpy as np
model = YOLO("yolov8l.pt")
font = cv2.FONT_HERSHEY_SIMPLEX
kamera = cv2.VideoCapture(0)

while True:

    ret, img = kamera.read()
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(rgb_img, verbose=False)
    labels= results[0].names

    for i in range(len(results[0].boxes)):
        x1, y1, x2, y2 = results[0].boxes.xyxy[i]
        score = results[0].boxes.conf[i]
        label = results[0].boxes.cls[i]
        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
        name = labels[label]
        if score < 0.5:
            continue

        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = name + ' ' + str(format(score, '.2f'))
        width= x2-x1
        height = y2 - y1
        # px to cm
        # witdh için : 7cm (seçilin telefon)

        # Ratio pixel to cm

        ratiow_px_cm = 150 / 8.5
        cmw = width / ratiow_px_cm

        ratioh_px_cm = 270 / 16
        cmh = height / ratioh_px_cm
        #cv2.putText(img, "w:{} h:{} cm".format(int(cmw),int(cmh)), (x1, y1 - 10), font, 0.7, (255, 0, 255), 2)
        cv2.putText(img, "w:{} ".format(cmw), (x1, y1 - 10), font, 0.7, (255, 0, 255), 2)

        cv2.putText(img, text, (x2, y2 - 10), font, 1.2, (255, 0, 255), 2)

    cv2.imshow("kamera", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
kamera.release()
cv2.destroyAllWindows()