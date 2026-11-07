from ultralytics import YOLO
import cv2
from yolo_segmentation import YOLOSegmentation

def main():

    cap = cv2.VideoCapture(0)
    #model = YOLO("yolov8m-seg.pt")
    ys = YOLOSegmentation("yolov8m-seg.pt")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        # Görüntüyü yeniden boyutlandır
        #frame = cv2.resize(frame, None, fx=0.7, fy=0.7)

        # Segmentasyonu algıla
        bboxes, classes, segmentations, scores = ys.detect(frame)

        # Sonuçları çiz
        for bbox, class_id, seg, score in zip(bboxes, classes, segmentations, scores):
            #result = ys.track(source=0, show=True, conf=0.5)
            (x, y, x2, y2) = bbox
            cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), 2)
            # cv2.putText(frame, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            #cv2.putText(frame, str(bbox), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            # cv2.polylines(frame, [seg], True, (255, 0, 0), 2)
            width= x2-x
            height= y2-y
            #px to cm
            #witdh için : 7cm (seçilin telefon)

            #Ratio pixel to cm

            ratiow_px_cm= 136/7
            cmw= width/ratiow_px_cm
            ratioh_px_cm = 270 /16
            cmh = height/ ratioh_px_cm

            cv2.putText(frame,"w:{} cm h:{} cm".format(cmw,cmh) , (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            #cv2.putText(frame,"height: {}".format(height) , (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        # Sonuçları göster
        cv2.imshow('frame', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()




if __name__=="__main__":
    main()
