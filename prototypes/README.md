# Prototip Ölçüm Denemeleri (Şubat – Mart 2024)

Bu klasör, final sistemdeki **piksel → santimetre ölçümüne** nasıl ulaşıldığını gösterir. Dosyaları kronolojik sırayla okumak projenin gelişimini anlatır. Final ve çalışan kod `app/camera_app/views.py` dosyasıdır. Buradaki dosyalar bu koda giden yoldaki denemelerdir.

Temel fikir dört aşamada da aynı:

```
oran (px/cm) = referans objenin piksel boyu / referans objenin gerçek boyu (cm)
gerçek boyut (cm) = ölçülen nesnenin piksel boyu / oran
```

Aşamalar arasında değişen şey nesnenin **nasıl bulunduğu**: önce kenar ve konturlarla, sonra YOLO kutusuyla, en son kendi eğittiğimiz modelle.

---

## 01 – Klasik kontur ölçümü (26.02.2024)
`01_klasik_kontur_olcumu/`

| Dosya | Açıklama |
|---|---|
| `size_object.py` | Tek fotoğraf üzerinde çalışır. Canny kenar bulma → kontur → `minAreaRect`. En soldaki nesne 2×2 cm referans kabul edilir ve diğer nesnelerin en ve boyu hesaplanır. |
| `webcam_kontur_olcumu.py` | Aynı yöntemi canlı webcam görüntüsüne uygular. |
| `KAYNAK_README.md` | Başlangıç noktası olarak kullanılan açık kaynak örnek: noorkhokhar99/Measure-size-of-objects-in-an-image-using-OpenCV |

**Neden bırakıldı:** Kontur yöntemi nesnenin **ne olduğunu** bilmez. Gölge, bant dokusu ya da ışık değişimi de kontur üretir ve ölçüm bozulur. Konveyörde ürünü hem tanımamız hem de ölçmemiz gerekiyordu.

## 02 – YOLO kutusu ile ölçüm, fındık denemesi (27.02.2024)
`02_yolo_kutu_olcumu_findik/`

| Dosya | Açıklama |
|---|---|
| `Nuts.pt` | Roboflow'da etiketlenmiş fındık görüntüleriyle eğitilen ilk özel YOLOv8 modeli |
| `foto_olcum.py` | `4.JPG` üzerinde fındıkları bulur ve kutu genişliğini `153 px = 14 mm` oranıyla cm'ye çevirir |
| `video_olcum.py` | Aynı fikri videoya taşıma denemesi. **Not:** Mask R-CNN örneğinden kalan `r["class_ids"]` söz dizimi yüzünden bu haliyle çalışmaz. Ara deneme olarak tutuldu. |

**Kazanım:** Nesne tespiti ile ölçüm ilk kez birleştirildi.

## 03 – Segmentasyon modeliyle webcam ölçümü (07–09.03.2024)
`03_segmentasyon_webcam_olcumu/`

| Dosya | Açıklama |
|---|---|
| `olcum_webcam_segmentasyon.py` | Ölçümün ilk "doğru biten" sürümü. Hazır `yolov8m-seg.pt` (COCO) ile webcam'de nesneyi bulur, **genişlik ve yükseklik** değerlerini cm olarak yazar. Referans olarak telefon kullanıldı: `136 px = 7 cm`, `270 px = 16 cm`. |
| `yolo_segmentation.py` | Yukarıdaki scriptin kullandığı yardımcı sınıf. Kaynağı pysource.com'daki YOLOv8 segmentation örneği. |
| `olcum_webcam_yolov8l.py` | Hazır `yolov8l.pt` tespit modeliyle webcam üzerinde ölçüm |
| `olcum_video_nuts.py` | Nuts.pt ile video üzerinde ölçüm (video dosyası klasöre alınmadı) |

`yolov8m-seg.pt` ve `yolov8l.pt` Ultralytics'in hazır modelleridir, ilk çalıştırmada otomatik indirilir. Bu yüzden klasöre eklenmedi.

**Neden bırakıldı:** Hazır COCO modeli ruj, krem kutusu gibi ürünlerimizi tanımıyordu. Bu nedenle 6 sınıflı kendi veri setimizi oluşturup `yolov8s` modelini eğittik (bkz. `training/`). Ölçüm için maske gerekmediği, kutu yeterli olduğu için segmentasyon yerine tespit modeli seçildi. Tespit modeli iki kamerayla gerçek zamanlı çalışırken de daha hızlıydı.

## 04 – Kalibrasyon aracı
`04_kalibrasyon/kalibrasyon.py`

Final sistemde her kamera için ayrı bir genişlik ve yükseklik oranı kullanılır (`views.py`):

```python
CAMERA1_RATIOW_PX_CM = 289 / 11.2   # üst kamera
CAMERA1_RATIOH_PX_CM = 253 / 9.2
CAMERA2_RATIOW_PX_CM = 277 / 11     # yan kamera
CAMERA2_RATIOH_PX_CM = 226 / 7
```

Bu değerler, proje sırasında referans obje ekranda ölçülüp elle hesaplanarak bulunmuştu. `kalibrasyon.py` aynı işlemi tekrarlanabilir hale getirmek için **sonradan eklenmiştir**. Kameralar yeniden kurulduğunda oranları yeniden hesaplamak için kullanılır:

```bash
# Üst kamera, elle seçim: SPACE ile kareyi dondur, objeyi fareyle seç, ENTER'a bas, q ile bitir
python kalibrasyon.py --camera 0 --width-cm 11.2 --height-cm 9.2

# Yan kamera (final sistemde 90° döndürülüyor)
python kalibrasyon.py --camera 1 --rotate --width-cm 11 --height-cm 7

# Otomatik: model referans objeyi bulur, 30 karenin medyanını alır
python kalibrasyon.py --camera 0 --width-cm 11.2 --height-cm 9.2 --model ../../app/27Epoch.pt
```

Script, sonuçta `views.py` dosyasına yapıştırılacak satırları ekrana yazar.

> **Sınırlılık:** Bu yöntem nesnelerin kameraya hep aynı uzaklıkta olduğunu varsayar. Bu yüzden kameralar bantta sabitlendi. Bunun sonucu olarak raporda paketleme için ±2 birimlik hata payı bırakılması önerildi. Daha hassas bir sistem için kamera iç parametre kalibrasyonu (satranç tahtası ile `cv2.calibrateCamera`) ve derinlik bilgisi eklenebilir.
