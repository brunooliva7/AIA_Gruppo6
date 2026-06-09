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
frame_linea_persa = 0
SOGLIA_ALLARME_LINEA_PERSA = 30  # Numero di frame consecutivi senza linea prima di lanciare l'allarme (circa 1 secondo a 30 FPS)
SOGLIA_AREA_INCROCIO = 10000  # Area minima per considerare un incrocio
testo_rilevato_global = "In attesa di cartelli..."
running = True
mappa_cartelli_global = {}  # Dizionario che conterrà { "TESTO": "DIREZIONE" }


# STATI DEL ROBOT
STATO_NAVIGAZIONE = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"

stato_attuale = STATO_NAVIGAZIONE
direzione_da_prendere = None  # "SINISTRA" o "DESTRA"
tempo_inizio_svolta = 0


# ==========================================
# 1. FUNZIONE OCR: ELABORAZIONE IN BACKGROUND
# ==========================================
def thread_elaborazione_ocr():
    global frame_condiviso, testo_rilevato_global, running
    print("[THREAD OCR] Avviato con successo.")
    
    while running:
        if frame_condiviso is None:
            time.sleep(0.1)
            continue
        
        img_ocr = frame_condiviso.copy()
        larghezza_frame = img_ocr.shape[1]
        gray = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2GRAY)
        
        try:
            risultati = lettore_ocr.readtext(gray)
            nuovi_cartelli = {}

            #Dividiamo lo schermo in 3 zone verticali: SINISTRA, CENTRO, DESTRA
            tre_zone = larghezza_frame // 3                                     # ES. 640px -> 213px per zona
            
            for (bbox, testo, probabilita) in risultati:
                testo_pulito = testo.strip().upper()
                print(f"[DEBUG OCR] Ho letto: '{testo_pulito}' | Confidenza: {probabilita:.2f}")
                
                if probabilita > 0.5 and len(testo_pulito) > 0:
                    # Calcoliamo il centro X del cartello usando il bounding box
                    # bbox[0][0] è la X in alto a sinistra, bbox[1][0] è la X in alto a destra
                    x_centro = int((bbox[0][0] + bbox[1][0]) / 2)

                    if x_centro < tre_zone:
                        direzione = "SINISTRA"
                    elif x_centro < 2 * tre_zone:
                        direzione = "DRITTO"
                    else:
                        direzione = "DESTRA"


                    nuovi_cartelli[testo_pulito] = direzione                    # CHIAVE : TESTO, VALORE : DIREZIONE

            risultati_ocr = nuovi_cartelli
            
            if nuovi_cartelli:
                mappa_cartelli_global = nuovi_cartelli


        except Exception as e:
            print(f"[ERRORE OCR] {e}")
            
        time.sleep(0.4)


# ==========================================
# 2. FUNZIONE LINEE GUIDA: ELABORAZIONE VISIVA
# ==========================================
def elabora_linee_guida(frame, blu_lower, blu_upper):
    """
    Isola le linee guida blu, calcola i momenti per determinare il centro,
    disegna gli indicatori visivi e stampa i comandi per i motori.
    """
    frame_linea_persa = 0
    altezza, larghezza, _ = frame.shape
    centro_camera = larghezza // 2
    incrocio_rilevato = False
    errore_x = 0
    
    # Pre-processing del frame per eliminare rumore
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blu_lower, blu_upper)
    
    # Calcolo dei momenti dell'immagine per trovare il centro della maschera
    moments = cv2.moments(mask)
    area_linea_guida = moments["m00"]

    #Se l'area diventa troppo grande, potrebbe essere un incrocio (molte linee blu)
    if area_linea_guida > SOGLIA_AREA_INCROCIO:
        incrocio_rilevato = True
        print("Incrocio rilevato!")

    if area_linea_guida > 0:
        frame_linea_persa = 0  # Reset del contatore se la linea è trovata
        centro_linea_guida_x = int(moments["m10"] / moments["m00"])
        centro_linea_guida_y = int(moments["m01"] / moments["m00"])
        errore_x = centro_linea_guida_x - centro_camera

        # Disegno del cerchio e della linea di allineamento
        cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
        cv2.line(frame, (centro_camera, altezza // 2), (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)

    else: 
        errore_x = None  # Nessuna linea trovata, non possiamo calcolare l'errore        

        # Se la linea guida è persa, incrementiamo il contatore e verifichiamo se è il caso di lanciare un allarme
        frame_linea_persa += 1
    
        # CASO 1: La linea non c'è, ma Pepper sta leggendo un cartello
        if "Nessun testo" not in testo_rilevato_global and testo_rilevato_global != "In attesa di cartelli...":
            print("Linea persa, ma sto leggendo un cartello. Rimango in attesa...")
            # Qui mandi un comando di STOP intenzionale a Pepper, non un errore.
            comando_motori = "STOP_LETTURA" 
        
        # CASO B: La linea è sparita e non c'è nessun cartello per troppo tempo
        elif frame_linea_persa > SOGLIA_ALLARME_LINEA_PERSA:
            cv2.putText(frame, "ERRORE CRITICO: LINEA PERDUTA", (10, 40), ...)
            comando_motori = "EMERGENCY_STOP"
        
    return frame, mask, incrocio_rilevato, errore_x



#==========================================
# 3. FUNZIONE CENTRALE: MACCHINA A STATI
# ==========================================
def gestisci_macchina_a_stati(errore_x, incrocio_rilevato):
    """
    Gestisce la logica decisionale e i comportamenti del robot basandosi 
    sullo stato attuale e sugli input sensoriali.
    """
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta
    
    # ----------------------------------------------------
    # COMPORTAMENTO: NAVIGAZIONE STANDARD
    # ----------------------------------------------------
    if stato_attuale == STATO_NAVIGAZIONE:
        if incrocio_rilevato:
            print("[STATO] Incrocio rilevato! Cambio stato: ATTESA SCELTA.")
            stato_attuale = STATO_INCROCIO_ATTESA
            print("Comando motori: STOP")  # Ferma Pepper all'incrocio
        else:
            if errore_x is not None:
                tolleranza = 50
                if errore_x > tolleranza:
                    print("Comando motori: Curva a Destra")
                elif errore_x < -tolleranza:
                    print("Comando motori: Curva a Sinistra")
                else:
                    print("Comando motori: Avanti Dritto")
            else:
                print("Comando motori: STOP (Linea temporaneamente perduta)")

    # ----------------------------------------------------
    # COMPORTAMENTO: ATTESA SCELTA UTENTE (VOCALE / TASTIERA)
    # ----------------------------------------------------
    elif stato_attuale == STATO_INCROCIO_ATTESA:
        # In questo stato i motori rimangono fermi.
        # Pepper aspetta l'assegnazione della variabile 'direzione_da_prendere' 
        # e il cambio di stato a 'STATO_SVOLTA_AUTOMATICA' effettuati dall'input utente.
        pass

    # ----------------------------------------------------
    # COMPORTAMENTO: MANOVRA DI SVOLTA AUTOMATICA / SUPERAMENTO
    # ----------------------------------------------------
    elif stato_attuale == STATO_SVOLTA_AUTOMATICA:
        if direzione_da_prendere == "SINISTRA":
            print("Comando motori: ROTAZIONE SUL POSTO A SINISTRA")
            durata_manovra = 2.5
        elif direzione_da_prendere == "DESTRA":
            print("Comando motori: ROTAZIONE SUL POSTO A DESTRA")
            durata_manovra = 2.5
        else:  # DRITTO
            print("Comando motori: AVANTI DRITTO (Superamento Incrocio)")
            durata_manovra = 1.5
            
        # Controllo temporale per terminare la manovra "cieca"
        if time.time() - tempo_inizio_svolta > durata_manovra:
            print("[STATO] Manovra completata. Ritorno in NAVIGAZIONE DI LINEA.")
            stato_attuale = STATO_NAVIGAZIONE


# ==========================================
# 4. FUNZIONE INTERFACCIA: OVERLAY GRAFICO OCR
# ==========================================
def applica_overlay_ocr(frame,stato, mappa_cartelli):
    cv2.putText(frame, f"STATO: {stato}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    if stato == STATO_INCROCIO_ATTESA:
        cv2.putText(frame, f"INCROCIO - CARTELLI TROVATI ({len(mappa_cartelli)}):", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        y_pos = 90
        for i, (testo, dir_associata) in enumerate(mappa_cartelli.items()):
            testo_display = f"Tasto [{i+1}]: {testo} -> ({dir_associata})"
            cv2.putText(frame, testo_display, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += 25
    elif stato == STATO_SVOLTA_AUTOMATICA:
        cv2.putText(frame, f"MANOVRA ATTIVA: {direzione_da_prendere}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
    return frame


# ==========================================
# 5. MAIN PROGRAM: FLUSSO PRINCIPALE
# ==========================================
def main():
    global frame_condiviso, mappa_cartelli_global, running
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta
    
    # Inizializzazione sorgente video
    video = cv2.VideoCapture(1)
    
    # Definizione range colore HSV per la linea blu
    BLU_LOWER = np.array([100, 150, 50])
    BLU_UPPER = np.array([140, 255, 255])
    
    # Avvio del thread OCR in modalità Daemon
    ocr_worker = threading.Thread(target=thread_elaborazione_ocr, daemon=True)
    ocr_worker.start()
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Errore nella lettura del video.")
            break
            
        # Aggiornamento costante del frame condiviso per l'asincronia dell'OCR
        frame_condiviso = frame.copy()

        # Copia istantanea per evitare conflitti di thread durante questo ciclo di rendering
        cartelli_disponibili = mappa_cartelli_global.copy()
        
        # 1. ELABORAZIONE FRAME (Moduli Visivi)
        frame_linee, maschera_linee, errore_x, incrocio_rilevato = elabora_linee_guida(frame.copy(), BLU_LOWER, BLU_UPPER)
        
        # 2. CHIAMATA ALLA MACCHINA A STATI (Logica Decisionale)
        gestisci_macchina_a_stati(errore_x, incrocio_rilevato)
        
        # 3. RENDERING GRAFICO
        frame_finale = applica_overlay_ocr(frame_linee, stato_attuale, cartelli_disponibili)
        cv2.imshow("Maschera", maschera_linee)
        cv2.imshow("Pepper Navigation System", frame_finale)
        
        # Interruzione con il tasto 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    # Chiusura pulita delle risorse
    running = False
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()