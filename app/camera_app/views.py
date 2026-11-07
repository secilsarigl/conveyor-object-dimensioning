# camera_app/views.py
# Final sürüm — BM496 final raporundaki (Şekil 5-10) kodlardan yeniden oluşturulmuştur.
# Raporda görünmeyen tek kısım "veritabanına kaydetme" bloğudur; aşağıda
# [YENİDEN YAZILDI] etiketiyle işaretlendi. Renk kanalı hatası [DÜZELTİLDİ] ile işaretli.
import os
import random
import time

import cv2
from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect, render
from ultralytics import YOLO

from .models import Measurement

# Model dosyası manage.py ile aynı klasörde durur (camera_project/27Epoch.pt)
MODEL_PATH = settings.BASE_DIR / "27Epoch.pt"

# Piksel/cm dönüşüm oranları — referans obje ile kalibre edildi (rapor 6.3).
# Kameraların konveyöre uzaklığı değişirse bu değerler yeniden ölçülmelidir.
CAMERA1_RATIOW_PX_CM = 289 / 11.2   # üst kamera, genişlik
CAMERA1_RATIOH_PX_CM = 253 / 9.2    # üst kamera, yükseklik
CAMERA2_RATIOW_PX_CM = 277 / 11     # yan kamera, genişlik
CAMERA2_RATIOH_PX_CM = 226 / 7      # yan kamera, yükseklik

# [DÜZELTİLDİ] Windows'ta varsayılan Media Foundation sürücüsü bazı kameralarda
# düz yeşil görüntü veriyor; DirectShow kullanılıyor.
CAMERA_BACKEND = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY

CONF_THRESHOLD = 0.5      # bu güven skorunun altındaki tespitler atlanır
NEW_PRODUCT_GAP_S = 2     # 2 sn tespit olmazsa sıradaki ürün yeni kabul edilir


def welcome(request):
    return render(request, 'camera_app/welcome.html')


def generate_unique_product_id():
    return random.randint(10000, 99999)


def save_product(product, tek_kamera=False):
    """[YENİDEN YAZILDI] İki kameradan da ölçüsü alınmış ürünü veritabanına yazar.

    tek_kamera=True ise (2. kamera açılamadığında, ör. masa üstü test) ürün
    yalnızca 1. kameranın ölçüsüyle kaydedilir; 2. kamera alanları 0 yazılır
    ve tabloda "-" olarak görünür.
    """
    if not product:
        return
    if "camera2_width" not in product and not tek_kamera:
        return
    Measurement.objects.create(
        product_id=product["product_id"],
        product_name=product["product_name"],
        camera1_width=product["camera1_width"],
        camera1_height=product["camera1_height"],
        camera2_width=product.get("camera2_width", 0),
        camera2_height=product.get("camera2_height", 0),
    )


@login_required
def process_cameras(request):
    if request.method == 'POST':
        if 'clear' in request.POST:
            Measurement.objects.all().delete()
            return redirect('process_cameras')

        if 'start' in request.POST:
            # YOLO modelini yükle
            model = YOLO(str(MODEL_PATH))

            # Kameraları başlat
            kamera1 = cv2.VideoCapture(0, CAMERA_BACKEND)   # üst kamera
            kamera2 = cv2.VideoCapture(1, CAMERA_BACKEND)   # yan kamera

            # [EKLENDİ] 2. kamera yoksa tek kamera test modu
            # (sanal kameralar 'açık' görünüp kare vermeyebilir, bu yüzden bir kare de denenir)
            tek_kamera = not kamera2.isOpened() or not kamera2.read()[0]
            if tek_kamera:
                print("UYARI: 2. kamera açılamadı -> tek kamera test modu (yalnızca kamera 1 ölçüleri kaydedilir)")

            font = cv2.FONT_HERSHEY_SIMPLEX

            # Geçici ürün bilgileri saklanacak dictionary
            current_product = None
            last_detection_time = None

            while True:
                # Birinci kameradan bir kare yakala
                ret1, img1 = kamera1.read()
                # İkinci kameradan bir kare yakala
                ret2, img2 = kamera2.read()

                if ret1:
                    # İlk kareyi YOLO modeline ver ve tespitleri yap
                    # [DÜZELTİLDİ] Ultralytics numpy görüntüyü BGR bekler; orijinal koddaki
                    # cvtColor(BGR2RGB) renk kanallarını ters çeviriyordu, kaldırıldı.
                    results1 = model(img1, verbose=False)
                else:
                    img1 = cv2.putText(cv2.UMat(480, 640, cv2.CV_8UC3, (0, 0, 255)), "Camera 1 Error",
                                       (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA).get()
                    results1 = []

                if ret2:
                    # İkinci kareyi 90 derece döndür
                    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
                    # İkinci kareyi YOLO modeline ver ve tespitleri yap
                    results2 = model(img2, verbose=False)   # [DÜZELTİLDİ] BGR olarak ver
                else:
                    img2 = cv2.putText(cv2.UMat(480, 640, cv2.CV_8UC3, (0, 0, 255)), "Camera 2 Error",
                                       (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA).get()
                    results2 = []

                current_time = time.time()

                # Birinci kameradan ürün bilgilerini al
                if results1:
                    for i in range(len(results1[0].boxes)):
                        x1, y1, x2, y2 = results1[0].boxes.xyxy[i]
                        score = results1[0].boxes.conf[i]
                        label = results1[0].boxes.cls[i]
                        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                        name = results1[0].names[label]
                        if score < CONF_THRESHOLD:
                            continue

                        width_px = x2 - x1
                        height_px = y2 - y1

                        width_cm = round(width_px / CAMERA1_RATIOW_PX_CM, 2)
                        height_cm = round(height_px / CAMERA1_RATIOH_PX_CM, 2)

                        cv2.rectangle(img1, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        text = f"{name} {score:.2f} ({width_cm}cm x {height_cm}cm)"
                        cv2.putText(img1, text, (x1, y1 - 10), font, 0.7, (255, 0, 255), 2)

                        if current_product is None or current_time - last_detection_time > NEW_PRODUCT_GAP_S:
                            # [YENİDEN YAZILDI] Önceki ürün tamamlandıysa kaydet
                            save_product(current_product, tek_kamera)
                            product_id = generate_unique_product_id()
                            current_product = {
                                "product_id": product_id,
                                "product_name": name,
                                "camera1_width": width_cm,
                                "camera1_height": height_cm,
                                "timestamp": current_time,
                            }
                        last_detection_time = current_time

                # İkinci kameradan ürün bilgilerini al
                if results2:
                    for i in range(len(results2[0].boxes)):
                        x1, y1, x2, y2 = results2[0].boxes.xyxy[i]
                        score = results2[0].boxes.conf[i]
                        label = results2[0].boxes.cls[i]
                        x1, y1, x2, y2, score, label = int(x1), int(y1), int(x2), int(y2), float(score), int(label)
                        name = results2[0].names[label]
                        if score < CONF_THRESHOLD:
                            continue

                        width_px = x2 - x1
                        height_px = y2 - y1

                        width_cm = round(width_px / CAMERA2_RATIOW_PX_CM, 2)
                        height_cm = round(height_px / CAMERA2_RATIOH_PX_CM, 2)

                        cv2.rectangle(img2, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        text = f"{name} {score:.2f} ({width_cm}cm x {height_cm}cm)"
                        cv2.putText(img2, text, (x1, y1 - 10), font, 0.7, (255, 0, 255), 2)

                        if current_product and current_product["product_name"] == name:
                            current_product["camera2_width"] = width_cm
                            current_product["camera2_height"] = height_cm

                # İki görüntüyü aynı yüksekliğe getirip yan yana göster
                if img1.shape != img2.shape:
                    height1 = min(img1.shape[0], img2.shape[0])
                    width1 = int(img1.shape[1] * (height1 / img1.shape[0]))
                    width2 = int(img2.shape[1] * (height1 / img2.shape[0]))

                    img1 = cv2.resize(img1, (width1, height1))
                    img2 = cv2.resize(img2, (width2, height1))

                combined_img = cv2.hconcat([img1, img2])
                cv2.imshow("Kameralar", combined_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # [YENİDEN YAZILDI] Döngüden çıkarken son ürünü de kaydet
            save_product(current_product, tek_kamera)

            kamera1.release()
            kamera2.release()
            cv2.destroyAllWindows()

    measurements = Measurement.objects.all()
    return render(request, 'camera_app/index.html', {'measurements': measurements})


def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # [DÜZELTİLDİ] Kayıt sonrası doğrudan ölçüm sayfasına git
            return redirect('process_cameras')
    else:
        form = UserCreationForm()
    return render(request, 'camera_app/register.html', {'form': form})
