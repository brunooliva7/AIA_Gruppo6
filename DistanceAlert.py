import cv2
import time
import os
from cvlib.object_detection import YOLO

# ---------------------------------------------------------
# 1. PARAMETRI DI CALIBRAZIONE
# Questo valore dipende dalla tua webcam. 
# Se la distanza calcolata è sbagliata, aumenta o diminuisci questo numero.
FOCAL_LENGTH = 543

# Distanza limite per far scattare l'allarme (in centimetri)
DISTANZA_ALLARME = 300 

ALTEZZE_REALI_CM = {
    'person': 170, 'dog': 60, 'cat': 30, 'bird': 15,
    'chair': 90, 'sofa': 85, 'bed': 60, 'dining table': 75,
    'toilet': 80, 'refrigerator': 180, 'tv': 65, 'laptop': 25,
    'mouse': 4, 'keyboard': 15, 'cell phone': 15, 'remote': 20,
    'clock': 30, 'microwave': 30, 'oven': 85, 'toaster': 20,
    'sink': 90, 'hair drier': 25, 'bottle': 25, 'wine glass': 20,
    'cup': 10, 'fork': 20, 'knife': 22, 'spoon': 15,
    'bowl': 10, 'book': 25, 'vase': 25, 'potted plant': 60,
    'backpack': 45, 'umbrella': 80, 'handbag': 30, 'tie': 45,
    'suitcase': 65, 'scissors': 15, 'teddy bear': 40, 'toothbrush': 20
}

ALTEZZA_DEFAULT = 50 

def calcola_distanza(altezza_reale_cm, altezza_pixel):
    if altezza_pixel == 0:
        return 0
    return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

def invia_alert(etichetta, distanza):
    print(f"⚠️ ALLARME! {etichetta} rilevato a soli {distanza:.2f} metri! ⚠️")

# ---------------------------------------------------------
# 2. FUNZIONE PRINCIPALE
# ---------------------------------------------------------
def main():
    cartella_script = os.path.dirname(os.path.abspath(__file__))
    cartella_yolo = os.path.join(cartella_script, "Yolo4-tiny") 

    yolo_weights = os.path.join(cartella_yolo, "yolov4-tiny.weights")
    yolo_cfg = os.path.join(cartella_yolo, "yolov4-tiny.cfg")
    coco_names = os.path.join(cartella_yolo, "coco.names")

    print("Caricamento rete neurale YOLOv4-Tiny...")
    try:
        yolo_model = YOLO(yolo_weights, yolo_cfg, coco_names)
    except Exception as e:
        print(f"Errore: Impossibile trovare i file in {cartella_yolo}.")
        return
    

    # Apri la webcam (0 è la webcam predefinita)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Errore critico: Impossibile accedere alla webcam.")
        return

    print("✅ Sistema avviato! Rilevamento in corso...")
    print("Premi 'q' sulla tastiera per uscire.")

    ultimo_alert = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Flusso video interrotto.")
            break

        # ESECUZIONE DELLA RETE NEURALE
        bbox, labels, conf = yolo_model.detect_objects(frame)

        for i, label in enumerate(labels):
            try:
                x_min, y_min, x_max, y_max = bbox[i]
                altezza_pixel = y_max - y_min
                
                # 1. Calcolo matematico con il nome esatto
                altezza_reale = ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT)
                distanza_cm = calcola_distanza(altezza_reale, altezza_pixel)
                distanza_m = distanza_cm / 100

                # 2. FILTRO SEMANTICO: Tutto diventa "PERSONA" o "OGGETTO"
                if label == 'person':
                    categoria_mostrata = "PERSONA"
                else:
                    categoria_mostrata = "OGGETTO"

                # 3. Logica di Allarme
                if distanza_cm < DISTANZA_ALLARME:
                    colore_box = (0, 0, 255) # Rosso
                    testo = f"ALLARME! {categoria_mostrata}: {distanza_m:.2f}m"
                    if time.time() - ultimo_alert > 1:
                        invia_alert(categoria_mostrata, distanza_m)
                        ultimo_alert = time.time()
                else:
                    colore_box = (0, 255, 0) # Verde
                    testo = f"{categoria_mostrata}: {distanza_m:.2f}m"

                # Disegno del box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colore_box, 2)
                (w_txt, h_txt), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x_min, y_min - 20), (x_min + w_txt, y_min), (0,0,0), -1)
                cv2.putText(frame, testo, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore_box, 2)
            
            except IndexError:
                pass

        cv2.imshow("Sistema Anti-Intrusione (Stabile)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Spegnimento sicuro
    cap.release()
    cv2.destroyAllWindows()
    print("Sistema chiuso correttamente.")

if __name__ == "__main__":
    main()