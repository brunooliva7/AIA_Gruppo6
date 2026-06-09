import cv2
import time
import os
import threading
from cvlib.object_detection import YOLO

# ---------------------------------------------------------
# 1. PARAMETRI DI CALIBRAZIONE E DIZIONARIO
# ---------------------------------------------------------
FOCAL_LENGTH = 300
DISTANZA_ALLARME = 150 

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
    if altezza_pixel == 0: return 0
    return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

def invia_alert(etichetta, distanza):
    print(f"⚠️ ALLARME! {etichetta} rilevato a soli {distanza:.2f} metri! ⚠️")

# ---------------------------------------------------------
# 2. MEMORIA CONDIVISA (Senza Queue)
# ---------------------------------------------------------
dati_condivisi = {
    "frame_da_analizzare": None,
    "bbox": [],
    "labels": [],
    "occupato": False,
    "in_esecuzione": True
}

# ---------------------------------------------------------
# 3. THREAD SECONDARIO (Il Lavoratore)
# ---------------------------------------------------------
def thread_rilevamento(yolo_model):
    global dati_condivisi
    print("🧠 Thread YOLO avviato. In attesa del video...")
    
    while dati_condivisi["in_esecuzione"]:
        # Se c'è un frame pronto e non sto già lavorando
        if dati_condivisi["frame_da_analizzare"] is not None and not dati_condivisi["occupato"]:
            
            # Alzo la "bandierina" per dire che sono occupato
            dati_condivisi["occupato"] = True
            
            # Copio il frame per non farlo sovrascrivere dal video in diretta
            frame_corrente = dati_condivisi["frame_da_analizzare"].copy()
            
            # Lavoro Pesante (YOLO)
            bbox, labels, conf = yolo_model.detect_objects(frame_corrente)
            
            # Salvo i risultati
            dati_condivisi["bbox"] = bbox
            dati_condivisi["labels"] = labels
            
            # Abbasso la "bandierina" per accettare un nuovo frame
            dati_condivisi["occupato"] = False
            
        # Pausa vitale (10 millisecondi). Senza questa, il thread prende il 100% della CPU
        time.sleep(0.01)

# ---------------------------------------------------------
# 4. FUNZIONE PRINCIPALE (Il Video)
# ---------------------------------------------------------
def main():
    global dati_condivisi

    cartella_script = os.path.dirname(os.path.abspath(__file__))
    cartella_yolo = os.path.join(cartella_script, "Yolo4-tiny") 

    yolo_weights = os.path.join(cartella_yolo, "yolov4-tiny.weights")
    yolo_cfg = os.path.join(cartella_yolo, "yolov4-tiny.cfg")
    coco_names = os.path.join(cartella_yolo, "coco.names")

    print("Caricamento rete neurale...")
    try:
        yolo_model = YOLO(yolo_weights, yolo_cfg, coco_names)
    except Exception as e:
        print(f"Errore caricamento YOLO: {e}")
        return

    # REGOLA D'ORO: Accendere la fotocamera PRIMA di avviare il Thread
    print("Accensione fotocamera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Errore critico: Impossibile accedere alla webcam.")
        return

    # Avvio del Thread (dopo che la webcam è al sicuro)
    t_rilevamento = threading.Thread(target=thread_rilevamento, args=(yolo_model,))
    t_rilevamento.start()

    print("✅ Sistema Multithread Puro Avviato! (Premi 'q' per uscire)")
    ultimo_alert = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Se YOLO ha finito il giro precedente, gli passo il frame nuovo
            if not dati_condivisi["occupato"]:
                dati_condivisi["frame_da_analizzare"] = frame.copy()

            # Leggo gli ultimi dati calcolati
            bbox = dati_condivisi["bbox"]
            labels = dati_condivisi["labels"]

            # Disegno e Calcolo
            for i, label in enumerate(labels):
                try:
                    x_min, y_min, x_max, y_max = bbox[i]
                    altezza_pixel = y_max - y_min
                    altezza_reale = ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT)
                    
                    distanza_cm = calcola_distanza(altezza_reale, altezza_pixel)
                    distanza_m = distanza_cm / 100

                    # FILTRO SEMANTICO
                    categoria_mostrata = "PERSONA" if label == 'person' else "OGGETTO"

                    if distanza_cm < DISTANZA_ALLARME:
                        colore_box = (0, 0, 255)
                        testo = f"ALLARME! {categoria_mostrata}: {distanza_m:.2f}m"
                        if time.time() - ultimo_alert > 1:
                            invia_alert(categoria_mostrata, distanza_m)
                            ultimo_alert = time.time()
                    else:
                        colore_box = (0, 255, 0)
                        testo = f"{categoria_mostrata}: {distanza_m:.2f}m"

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colore_box, 2)
                    (w_txt, h_txt), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x_min, y_min - 20), (x_min + w_txt, y_min), (0,0,0), -1)
                    cv2.putText(frame, testo, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore_box, 2)
                
                except IndexError:
                    pass

            cv2.imshow("Sistema Anti-Intrusione (Thread Base)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Chiusura pulita garantita
        print("Spegnimento in corso...")
        dati_condivisi["in_esecuzione"] = False
        t_rilevamento.join() 
        cap.release()
        cv2.destroyAllWindows()
        print("Sistema chiuso correttamente.")

if __name__ == "__main__":
    main()