import cv2
import numpy as np
import threading
import time
import os
import easyocr
from cvlib.object_detection import YOLO

# ==========================================
# 1. PARAMETRI E VARIABILI GLOBALI
# ==========================================
# Parametri YOLO
FOCAL_LENGTH = 300
DISTANZA_ALLARME = 150 
ALTEZZE_REALI_CM = {'person': 170, 'chair': 90, 'bottle': 25, 'backpack': 45} # Ridotto per brevità
ALTEZZA_DEFAULT = 50 

# Parametri Linee e OCR
BLU_LOWER = np.array([100, 150, 50])
BLU_UPPER = np.array([140, 255, 255])
SOGLIA_AREA_INCROCIO = 10000
SOGLIA_ALLARME_LINEA_PERSA = 30

# Stati Robot
STATO_NAVIGAZIONE = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"
stato_attuale = STATO_NAVIGAZIONE
direzione_da_prendere = None
tempo_inizio_svolta = 0

# Memoria Condivisa tra i Thread
dati_condivisi = {
    "frame_da_analizzare": None, # Frame inviato a YOLO e OCR
    "running": True,
    # Dati YOLO
    "yolo_bbox": [], "yolo_labels": [], "yolo_occupato": False,
    # Dati OCR
    "ocr_cartelli": {}, "ocr_occupato": False, "frame_linea_persa": 0
}

# ==========================================
# 2. I THREAD WORKER (Background)
# ==========================================
def thread_yolo(yolo_model):
    global dati_condivisi
    print("🧠 [THREAD] YOLO (Ostacoli) avviato.")
    while dati_condivisi["running"]:
        if dati_condivisi["frame_da_analizzare"] is not None and not dati_condivisi["yolo_occupato"]:
            dati_condivisi["yolo_occupato"] = True
            frame_corrente = dati_condivisi["frame_da_analizzare"].copy()
            
            bbox, labels, conf = yolo_model.detect_objects(frame_corrente)
            
            dati_condivisi["yolo_bbox"] = bbox
            dati_condivisi["yolo_labels"] = labels
            dati_condivisi["yolo_occupato"] = False
        time.sleep(0.01)

def thread_ocr(lettore_ocr):
    global dati_condivisi
    print("👁️ [THREAD] EasyOCR (Segnaletica) avviato.")
    while dati_condivisi["running"]:
        if dati_condivisi["frame_da_analizzare"] is not None and not dati_condivisi["ocr_occupato"]:
            dati_condivisi["ocr_occupato"] = True
            img_ocr = dati_condivisi["frame_da_analizzare"].copy()
            larghezza_frame = img_ocr.shape[1]
            gray = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2GRAY)
            
            try:
                risultati = lettore_ocr.readtext(gray)
                nuovi_cartelli = {}
                tre_zone = larghezza_frame // 3
                
                for (bbox, testo, probabilita) in risultati:
                    testo_pulito = testo.strip().upper()
                    if probabilita > 0.5 and len(testo_pulito) > 0:
                        x_centro = int((bbox[0][0] + bbox[1][0]) / 2)
                        if x_centro < tre_zone: direzione = "SINISTRA"
                        elif x_centro < 2 * tre_zone: direzione = "DRITTO"
                        else: direzione = "DESTRA"
                        nuovi_cartelli[testo_pulito] = direzione
                
                if nuovi_cartelli:
                    dati_condivisi["ocr_cartelli"] = nuovi_cartelli
            except Exception as e:
                pass
            
            dati_condivisi["ocr_occupato"] = False
        time.sleep(0.3) # L'OCR è pesante, gira meno frequentemente

# ==========================================
# 3. FUNZIONI DI ELABORAZIONE VISIVA
# ==========================================
def calcola_distanza(altezza_reale_cm, altezza_pixel):
    if altezza_pixel == 0: return 0
    return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

def elabora_linee_guida(frame, blu_lower, blu_upper):
    altezza, larghezza, _ = frame.shape
    centro_camera = larghezza // 2
    incrocio_rilevato = False
    errore_x = None
    
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blu_lower, blu_upper)
    moments = cv2.moments(mask)
    area_linea_guida = moments["m00"]

    if area_linea_guida > SOGLIA_AREA_INCROCIO:
        incrocio_rilevato = True

    if area_linea_guida > 0:
        dati_condivisi["frame_linea_persa"] = 0
        centro_linea_guida_x = int(moments["m10"] / moments["m00"])
        centro_linea_guida_y = int(moments["m01"] / moments["m00"])
        errore_x = centro_linea_guida_x - centro_camera

        cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
        cv2.line(frame, (centro_camera, altezza // 2), (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)
    else:
        dati_condivisi["frame_linea_persa"] += 1
        if dati_condivisi["frame_linea_persa"] > SOGLIA_ALLARME_LINEA_PERSA:
            cv2.putText(frame, "ERRORE CRITICO: LINEA PERDUTA", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    return frame, mask, incrocio_rilevato, errore_x

def gestisci_macchina_a_stati(errore_x, incrocio_rilevato, ostacolo_vicino):
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta
    
    # Priorità Assoluta: Se c'è un ostacolo vicino, i motori si fermano indipendentemente dallo stato
    if ostacolo_vicino:
        print("[MOTORI] EMERGENZA: STOP (Ostacolo rilevato)")
        return

    if stato_attuale == STATO_NAVIGAZIONE:
        if incrocio_rilevato:
            stato_attuale = STATO_INCROCIO_ATTESA
            print("[MOTORI] STOP (Incrocio Rilevato)")
        elif errore_x is not None:
            if errore_x > 50: print("[MOTORI] Curva a Destra")
            elif errore_x < -50: print("[MOTORI] Curva a Sinistra")
            else: print("[MOTORI] Avanti Dritto")

    elif stato_attuale == STATO_INCROCIO_ATTESA:
        pass # In attesa di input per cambiare stato

    elif stato_attuale == STATO_SVOLTA_AUTOMATICA:
        if time.time() - tempo_inizio_svolta > 2.5:
            stato_attuale = STATO_NAVIGAZIONE

# ==========================================
# 4. ENTRYPOINT: IL MAIN LOOP
# ==========================================
def main():
    global dati_condivisi, stato_attuale

    print("--- INIZIALIZZAZIONE SISTEMA PEPPER ---")
    
    # 1. Caricamento YOLO
    print("1/3 Caricamento Rete Neurale YOLO...")
    cartella_yolo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yolo4-tiny") 
    yolo_model = YOLO(os.path.join(cartella_yolo, "yolov4-tiny.weights"), 
                      os.path.join(cartella_yolo, "yolov4-tiny.cfg"), 
                      os.path.join(cartella_yolo, "coco.names"))

    # 2. Caricamento EasyOCR
    print("2/3 Inizializzazione motore OCR (può richiedere qualche secondo)...")
    lettore_ocr = easyocr.Reader(['it', 'en'], gpu=False)

    # 3. Accensione Telecamera
    print("3/3 Accensione Telecamera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Errore critico telecamera.")
        return

    # Avvio Thread
    threading.Thread(target=thread_yolo, args=(yolo_model,), daemon=True).start()
    threading.Thread(target=thread_ocr, args=(lettore_ocr,), daemon=True).start()

    print("✅ SISTEMA OPERATIVO. Premi 'q' per uscire.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            # Inviamo il frame ai thread se hanno finito di calcolare il precedente
            if not dati_condivisi["yolo_occupato"] and not dati_condivisi["ocr_occupato"]:
                dati_condivisi["frame_da_analizzare"] = frame.copy()

            ostacolo_vicino = False

            # Disegno YOLO (Ostacoli)
            for i, label in enumerate(dati_condivisi["yolo_labels"]):
                try:
                    x_min, y_min, x_max, y_max = dati_condivisi["yolo_bbox"][i]
                    alt_pixel = y_max - y_min
                    distanza_cm = calcola_distanza(ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT), alt_pixel)
                    
                    cat = "PERSONA" if label == 'person' else "OGGETTO"
                    if distanza_cm < DISTANZA_ALLARME:
                        ostacolo_vicino = True
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                        cv2.putText(frame, f"ALLARME {cat} ({distanza_cm/100:.1f}m)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(frame, f"{cat} ({distanza_cm/100:.1f}m)", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except IndexError: pass

            # Disegno Linee e Logica di Navigazione
            frame, maschera, incrocio, errore_x = elabora_linee_guida(frame, BLU_LOWER, BLU_UPPER)
            gestisci_macchina_a_stati(errore_x, incrocio, ostacolo_vicino)

            # Disegno Interfaccia Macchina a Stati / OCR
            cv2.putText(frame, f"STATO: {stato_attuale}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if stato_attuale == STATO_INCROCIO_ATTESA:
                cv2.putText(frame, f"CARTELLI RILEVATI:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                y_pos = 85
                for testo, dir_assoc in dati_condivisi["ocr_cartelli"].items():
                    cv2.putText(frame, f"- {testo} -> {dir_assoc}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    y_pos += 25

            cv2.imshow("Pepper - Visione Integrata", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Spegnimento in corso...")
        dati_condivisi["running"] = False
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()