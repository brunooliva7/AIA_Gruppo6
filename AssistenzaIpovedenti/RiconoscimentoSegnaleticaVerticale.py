import cv2
import numpy as np
import threading
import time
import easyocr

# Inizializziamo EasyOCR per l'Italiano e l'Inglese (esegue il download alla prima corsa)
print("[INFO] Inizializzazione motore OCR in corso...")
lettore_ocr = easyocr.Reader(['it', 'en'], gpu=False) # Metti True se hai una GPU Nvidia

# Variabili condivise tra i thread
frame_condiviso = None
testo_rilevato_global = "In attesa di cartelli..."
running = True

# La stanza che l'utente vuole raggiungere (es. ricevuta da Dialogflow)
STANZA_TARGET = "LABORATORIO DIEM"

# ==========================================
# THREAD 2: ELABORAZIONE OCR IN BACKGROUND
# ==========================================
def thread_elaborazione_ocr():
    global frame_condiviso, testo_rilevato_global, running
    
    print("[THREAD OCR] Avviato con successo.")
    
    while running:
        if frame_condiviso is None:
            time.sleep(0.1)
            continue
        
        img_ocr = frame_condiviso.copy()
        
        # Ottimizzazione: Convertiamo in scala di grigi per velocizzare l'OCR
        gray = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2GRAY)
        
        try:
            # EasyOCR analizza l'immagine (operazione pesante da 1-2 secondi)
            risultati = lettore_ocr.readtext(gray)
            
            # Analizziamo i testi trovati
            for (bbox, testo, probabilita) in risultati:
                testo_pulito = testo.strip().upper()

                # STAMPA DI TRACCIAMENTO (Guarda il terminale!)
                print(f"[DEBUG OCR] Ho letto: '{testo_pulito}' | Confidenza: {probabilita:.2f} | Cerco: '{STANZA_TARGET}'")
                
                # 1. Filtro di sicurezza: la stringa non deve essere vuota e la confidenza deve essere accettabile
                if probabilita > 0.5 and len(testo_pulito) > 0:
                    testo_rilevato_global = f"{testo_pulito} ({probabilita*100:.0f}%)"
                    
                    # 2. Controllo stringhe vuote per evitare falsi positivi radicali
                    if not STANZA_TARGET or STANZA_TARGET.isspace():
                        print("[ATTENZIONE] La variabile STANZA_TARGET è vuota! Il controllo è stato bloccato.")
                        continue
                    
                    # 3. Il confronto effettivo
                    if STANZA_TARGET in testo_pulito:
                        print(f"\n[!!!] COMBINAZIONE TROVATA [!!!]")
                        print(f"La parola '{STANZA_TARGET}' è dentro '{testo_pulito}'")
                        print(f"[Webhook inviato per Pepper]\n")


        except Exception as e:
            print(f"[ERRORE OCR] {e}")
    
    # Facciamo riposare il thread per non sovraccaricare la CPU al 100%
    time.sleep(0.5)

def main():
    global frame_condiviso, testo_rilevato_global, running
    
    # Colleghiamo Iriun Webcam (Usa l'indice corretto che hai testato prima, es. 1 o 2)
    video = cv2.VideoCapture(1)
    
    BLU_LOWER = np.array([100, 150, 50])
    BLU_UPPER = np.array([140, 255, 255])
    
    # Avviamo il thread dell'OCR in modalità "Daemon" (si chiude da solo quando chiudiamo il main)
    ocr_worker = threading.Thread(target=thread_elaborazione_ocr, daemon=True)
    ocr_worker.start()
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Errore nella lettura del video da Iriun.")
            break
            
        # Aggiorniamo il frame condiviso che il thread OCR andrà a leggere
        rame_condiviso = frame.copy()
        
        # --- PARTE 1: ELABORAZIONE LINEE GUIDA (Gira a 30 FPS fluidi) ---
        altezza, larghezza, _ = frame.shape
        centro_camera_x = larghezza // 2
        
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, BLU_LOWER, BLU_UPPER)
        
        moments = cv2.moments(mask)
        if moments["m00"] > 0:
            centro_linea_guida_x = int(moments["m10"] / moments["m00"])
            centro_linea_guida_y = int(moments["m01"] / moments["m00"])
            
            # Disegni grafici
            cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
            cv2.line(frame, (centro_camera_x, altezza // 2), (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)
        
        # --- PARTE 2: RENDERING GRAFICO ---
        # Sovrascriviamo sul video l'ultimo testo letto in background dal thread OCR
        cv2.putText(frame, f"Target cercato: {STANZA_TARGET}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"OCR Live: {testo_rilevato_global}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Navigazione Robot + OCR", frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    # Chiusura pulita
    running = False
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()