"""Kamera teşhis aracı: her kamera için iki sürücüyü dener, görüntülerin ortalama
rengini yazar ve kareleri kamera_test_*.jpg olarak kaydeder.
Kullanım: python kamera_test.py"""
import cv2

backends = {"DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}
for idx in (0, 1):
    for name, be in backends.items():
        cap = cv2.VideoCapture(idx, be)
        ok, frame = False, None
        for _ in range(15):              # ilk kareler boş/yeşil gelebilir, ısınma
            ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"kamera {idx} | {name:5s} | AÇILAMADI")
            continue
        b, g, r = frame.reshape(-1, 3).mean(axis=0)
        durum = "YEŞİL/BOŞ" if g > 100 and r < 30 and b < 30 else "görüntü var"
        dosya = f"kamera_test_{idx}_{name}.jpg"
        cv2.imwrite(dosya, frame)
        print(f"kamera {idx} | {name:5s} | {frame.shape[1]}x{frame.shape[0]} | ortalama BGR=({b:.0f},{g:.0f},{r:.0f}) -> {durum} | {dosya}")
