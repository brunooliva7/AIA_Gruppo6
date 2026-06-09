import cv2
import os
from cvlib.object_detection import YOLO  # Importante: importiamo la classe YOLO

# 1. SETUP DEI PERCORSI (Fuori dal ciclo video per non rallentare)
cartella_script = os.path.dirname(os.path.abspath(__file__))
cartella_yolo = os.path.join(cartella_script, "Yolo4-tiny")

yolo_weights = os.path.join(cartella_yolo, "yolov4-tiny.weights")
yolo_cfg = os.path.join(cartella_yolo, "yolov4-tiny.cfg")
coco_names = os.path.join(cartella_yolo, "coco.names")

# 2. CARICAMENTO DEL MODELLO LOCALE
print("Caricamento della rete neurale dalla cartella Yolo4-tiny...")
try:
    yolo_model = YOLO(yolo_weights, yolo_cfg, coco_names)
except Exception as e:
    print(f"Errore nel caricamento dei file: {e}")
    exit()

# 3. AVVIO WEBCAM
cap = cv2.VideoCapture(0)

print("🎯 Modalità Calibrazione Avviata.")
print("Mettiti a una distanza precisa (es. 2 metri) e guarda il numero rosso sullo schermo.")
print("Premi 'q' per uscire.")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # 4. RILEVAMENTO VELOCE (usando il nostro modello, non quello di default)
    bbox, labels, conf = yolo_model.detect_objects(frame)

    for i, label in enumerate(labels):
        # Facciamo il calcolo SOLO se l'oggetto è una persona
        if label == 'person':
            x_min, y_min, x_max, y_max = bbox[i]
            
            # IL CALCOLO DELL'ALTEZZA DEL RETTANGOLO
            altezza_pixel = y_max - y_min
            
            # Disegna il rettangolo verde
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            # Scrive l'altezza in pixel bella grande sotto al rettangolo
            testo = f"ALTEZZA: {altezza_pixel} px"
            cv2.putText(frame, testo, (x_min, y_max + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Strumento di Calibrazione", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()