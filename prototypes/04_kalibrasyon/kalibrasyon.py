"""
Kamera kalibrasyonu: piksel/cm oranını hesaplar.

Final sistemdeki (camera_app/views.py) şu sabitleri üretmek için kullanılır:
    CAMERA1_RATIOW_PX_CM, CAMERA1_RATIOH_PX_CM   (üst kamera)
    CAMERA2_RATIOW_PX_CM, CAMERA2_RATIOH_PX_CM   (yan kamera)

Mantık (rapor 6.3): Boyutu cetvelle ölçülmüş bir referans objeyi konveyörde,
ürünlerin geçeceği yere koy. Kamera objeyi kaç piksel görüyorsa, onu gerçek
boyuta bölünce "1 cm kaç piksel" oranı çıkar:
    oran = piksel_boyu / cm_boyu      ->   gercek_cm = piksel_boyu / oran
Kamera ile bant arasındaki mesafe değişirse oran da değişir; kamera her
yerinden oynadığında kalibrasyon tekrarlanmalıdır.

İki mod var:
  1) Elle seçim (varsayılan): SPACE ile kareyi dondur, fareyle objenin
     etrafına dikdörtgen çiz, ENTER'a bas.
  2) Otomatik (--model verilirse): YOLO modeli objeyi bulur, art arda
     birkaç karedeki kutu boyutlarının ortalaması alınır.

Örnekler:
  python kalibrasyon.py --camera 0 --width-cm 11.2 --height-cm 9.2
  python kalibrasyon.py --camera 1 --rotate --width-cm 11 --height-cm 7
  python kalibrasyon.py --camera 0 --width-cm 11.2 --height-cm 9.2 --model ../../app/27Epoch.pt
"""
import argparse
import statistics

import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Piksel/cm kalibrasyonu")
    p.add_argument("--camera", type=int, default=0, help="Kamera indeksi (0, 1, ...)")
    p.add_argument("--width-cm", type=float, required=True, help="Referans objenin gerçek genişliği (cm)")
    p.add_argument("--height-cm", type=float, required=True, help="Referans objenin gerçek yüksekliği (cm)")
    p.add_argument("--rotate", action="store_true",
                   help="Görüntüyü 90° saat yönünde döndür (final sistemde yan kamera böyle)")
    p.add_argument("--model", help="Verilirse YOLO ile otomatik ölçüm yapılır (.pt dosyası)")
    p.add_argument("--frames", type=int, default=30, help="Otomatik modda ortalaması alınacak kare sayısı")
    p.add_argument("--conf", type=float, default=0.5, help="Otomatik modda en düşük güven skoru")
    return p.parse_args()


def read_frame(cap, rotate):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("Kameradan görüntü alınamadı. --camera indeksini kontrol et.")
    if rotate:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def manual_mode(cap, args):
    widths, heights = [], []
    print("SPACE: kareyi dondur ve objeyi seç | q: bitir")
    while True:
        frame = read_frame(cap, args.rotate)
        view = frame.copy()
        cv2.putText(view, "SPACE: sec   q: bitir   olcum sayisi: %d" % len(widths),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            x, y, w, h = cv2.selectROI("Kalibrasyon", frame, showCrosshair=True)
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
                print(f"  ölçüm {len(widths)}: {w} px x {h} px")
    return widths, heights


def auto_mode(cap, args):
    from ultralytics import YOLO
    model = YOLO(args.model)
    widths, heights = [], []
    print(f"Otomatik mod: {args.frames} karede referans obje aranıyor...")
    while len(widths) < args.frames:
        frame = read_frame(cap, args.rotate)
        result = model(frame, verbose=False)[0]   # numpy görüntü BGR olarak verilir
        if len(result.boxes):
            i = int(result.boxes.conf.argmax())
            if float(result.boxes.conf[i]) >= args.conf:
                x1, y1, x2, y2 = map(int, result.boxes.xyxy[i])
                widths.append(x2 - x1)
                heights.append(y2 - y1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "olcum: %d/%d" % (len(widths), args.frames),
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Kalibrasyon", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    return widths, heights


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.camera)
    try:
        if args.model:
            widths, heights = auto_mode(cap, args)
        else:
            widths, heights = manual_mode(cap, args)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not widths:
        raise SystemExit("Hiç ölçüm alınmadı.")

    w_px = statistics.median(widths)
    h_px = statistics.median(heights)
    print("\n--- SONUÇ ---")
    print(f"Ölçüm sayısı      : {len(widths)}")
    print(f"Genişlik (medyan) : {w_px:.0f} px  -> oran = {w_px / args.width_cm:.3f} px/cm")
    print(f"Yükseklik (medyan): {h_px:.0f} px  -> oran = {h_px / args.height_cm:.3f} px/cm")
    if len(widths) > 1:
        print(f"Genişlik sapması  : ±{statistics.pstdev(widths):.1f} px")
    print("\nviews.py'a yapıştırılacak satırlar (kamera numarasını düzelt):")
    print(f"CAMERAx_RATIOW_PX_CM = {w_px:.0f} / {args.width_cm}")
    print(f"CAMERAx_RATIOH_PX_CM = {h_px:.0f} / {args.height_cm}")


if __name__ == "__main__":
    main()
