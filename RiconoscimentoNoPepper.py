import os

import cv2
import numpy as np
import threading
import time
import easyocr
import queue
import speech_recognition as sr
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


# Parametri Linee e OCR
#BLU_LOWER = np.array([100, 150, 50])
#BLU_UPPER = np.array([140, 255, 255])

#Parametri per condizioni di luce variabili (es. esterno con sole, interno con ombre)
BLU_LOWER = np.array([95, 70, 30])
BLU_UPPER = np.array([135, 255, 255])

# Configurazione Soglie Visive
SOGLIA_MINIMA_PIXEL = 3000       # Sotto questa quota di pixel blu, la linea è considerata persa
SOGLIA_Y_INCROCIO = 320          # Se il baricentro blu scende troppo in basso (es. > 320 su 480 di altezza), l'incrocio è vicino

SOGLIA_ALLARME_LINEA_PERSA = 30  # Numero di frame consecutivi senza linea prima di lanciare un allarme critico

# STATI DELL'ASSISTENTE INDOSSABILE
STATO_NAVIGAZIONE = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"

# Variables condivise tra i thread
dati_condivisi = {
    "frame_da_analizzare": None,    # Frame inviato a YOLO e OCR
    "running": True,

    #Dati YOLO
    "yolo_occupato": False,         # Flag per evitare che più thread accedano a YOLO contemporaneamente
    "yolo_bbox": [],                # Ultimi bounding box rilevati da YOLO
    "yolo_labels": [],              # Ultime etichette rilevate da YOLO 

    # Dati OCR
    "ocr_cartelli": {}, "ocr_occupato": False, "frame_linea_persa": 0
    
}

mappa_cartelli_global = {}  # Conterrà { "TESTO": "DIREZIONE" }
testo_rilevato_global = "In attesa di cartelli..."

stato_attuale = STATO_NAVIGAZIONE
direzione_da_prendere = None  
tempo_inizio_svolta = 0

# Gestione Coda Vocale asincrona (Evita che la telecamera scatti mentre il PC parla)
coda_voce = queue.Queue()
ultimo_messaggio_navigazione = ""
ultimo_tempo_voce = 0
INTERVALLO_VOCE = 1.8  # Secondi di silenzio obbligatori tra indicazioni di linea ripetitive



# ==========================================
# 1. THREAD YOLO: RILEVAMENTO OSTACOLI IN TEMPO REALE
# ==========================================

def thread_yolo(yolo_model):
    global dati_condivisi
    print("[THREAD] YOLO (Ostacoli) avviato.")
    while dati_condivisi["running"]:
        if dati_condivisi["frame_da_analizzare"] is not None and not dati_condivisi["yolo_occupato"]:
            dati_condivisi["yolo_occupato"] = True
            frame_corrente = dati_condivisi["frame_da_analizzare"].copy()
            
            bbox, labels, conf = yolo_model.detect_objects(frame_corrente)
            
            dati_condivisi["yolo_bbox"] = bbox
            dati_condivisi["yolo_labels"] = labels
            dati_condivisi["yolo_conf"] = conf
            dati_condivisi["yolo_occupato"] = False
        time.sleep(0.01)


# ==========================================
# 2. THREAD OCR: RILEVAMENTO TESTI CARTELLI IN TEMPO REALE
# ==========================================
def thread_elaborazione_ocr(lettore_ocr):
    global frame_condiviso, mappa_cartelli_global, running
    print("[THREAD OCR] Analisi cartelli attiva.")
    
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
                    
                    if probabilita > 0.4 and len(testo_pulito) > 1:
                        # Troviamo la X centrale del blocco di testo rilevato
                        x_centro = int((bbox[0][0] + bbox[1][0]) / 2)

                        # --- 1. CONTROLLO SEMANTICO (Priorità al significato del testo) ---
                        if "DESTRA" in testo_pulito:
                            direzione = "DESTRA"
                        elif "SINISTRA" in testo_pulito:
                            direzione = "SINISTRA"
                        elif "DRITTO" in testo_pulito or "AVANTI" in testo_pulito:
                            direzione = "DRITTO"

                        nuovi_cartelli[testo_pulito] = direzione 
                
                if nuovi_cartelli:
                    dati_condivisi["ocr_cartelli"] = nuovi_cartelli

            except Exception as e:
                print(f"[ERRORE OCR] {e}")
                
            dati_condivisi["ocr_occupato"] = False
        time.sleep(0.3) # L'OCR è pesante, gira meno frequentemente


# ======================================================
# 3. FUNZIONI DI ELABORAZIONE VISIVA DISTANZA E ALLARMI
# ======================================================
def calcola_distanza(altezza_reale_cm, altezza_pixel):
    if altezza_pixel == 0: return 0
    return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

def invia_alert(etichetta, distanza):
    print(f" ALLARME! {etichetta} rilevato a soli {distanza:.2f} metri! ")



# ======================================================
# 4. THREAD VOCE (OUTPUT): SINTESI VOCALE PC -> UTENTE
# ======================================================
def thread_notifiche_vocali():
    import pyttsx3
    print("[THREAD OUTPUT VOCALE] Pronto.")
    while dati_condivisi["running"]:
        try:
            # Preleva il messaggio dalla coda
            messaggio = coda_voce.get(timeout=0.5)
            print(f"\n>>> [GUIDA VOCALE]: {messaggio} <<<\n")
            
            # SOLUZIONE: Crea l'engine QUI, fresco per questo specifico messaggio
            engine = pyttsx3.init()
            engine.say(messaggio)
            engine.runAndWait()
            
            # Forza la pulizia dell'istanza per liberare il motore del OS
            del engine 
            
            coda_voce.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERRORE VOCALE] {e}")

def invia_voce(testo, prioritario=False):
    global ultimo_messaggio_navigazione, ultimo_tempo_voce
    tempo_attuale = time.time()

    if prioritario:
        # Svuota la coda da messaggi residui
        while not coda_voce.empty():
            try:
                coda_voce.get_nowait()
                coda_voce.task_done()
            except queue.Empty:
                break
        coda_voce.put(testo)
        ultimo_messaggio_navigazione = ''  # Resetta il messaggio di navigazione per evitare blocchi
        ultimo_tempo_voce = 0   # ← AGGIUNTO: resetta anche il timer
    else:
        if testo != ultimo_messaggio_navigazione or (tempo_attuale - ultimo_tempo_voce) > INTERVALLO_VOCE:
            # NON mettere in coda se c'è già un messaggio in attesa
            coda_voce.put(testo)
            ultimo_messaggio_navigazione = testo
            ultimo_tempo_voce = tempo_attuale


# ==========================================
# 6. FUNZIONE ELABORAZIONE LINEE GUIDA BLU
# ==========================================
def elabora_linee_guida(frame, blu_lower, blu_upper):

    """
    Isola le linee guida blu, calcola i momenti per determinare il centro,
    disegna gli indicatori visivi e stampa i comandi per i motori.
    """

    altezza, larghezza, _ = frame.shape
    centro_camera = larghezza // 2
    incrocio_rilevato = False
    errore_x = 0
    
    # Pre-processing del frame per eliminare rumore
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 2. GESTIONE LUCE ARTIFICIALE - SEPARAZIONE DEI CANALI ED APPLICAZIONE DI CLAHE SUL CANALE V (LUMINOSITÀ)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    v_normalizzato = clahe.apply(v)
    
    # Ricombiniamo i canali con la luminosità stabilizzata
    hsv_stabilizzato = cv2.merge([h, s, v_normalizzato])


    mask = cv2.inRange(hsv, blu_lower, blu_upper)


    
    # Calcolo dei momenti dell'immagine per trovare il centro della maschera
    # L'area (m00) ci dà un'indicazione di quanto è grande la regione blu rilevata. 
    # Se è troppo grande, potrebbe indicare un incrocio con molte linee blu.
    moments = cv2.moments(mask)
    area_linea_guida = moments["m00"]

    if area_linea_guida > SOGLIA_MINIMA_PIXEL:
        dati_condivisi["frame_linea_persa"] = 0
        centro_linea_guida_x = int(moments["m10"] / moments["m00"])
        centro_linea_guida_y = int(moments["m01"] / moments["m00"])
        errore_x = centro_linea_guida_x - centro_camera

        # --- LOGICA DI RILEVAMENTO INCROCIO PER LINEE INTERROTTE ---
        # Se la linea si interrompe davanti o il suo baricentro cade molto in basso,
        # significa che siamo arrivati alla sosta dell'incrocio visibile in foto.
        if centro_linea_guida_y > SOGLIA_Y_INCROCIO:
            incrocio_rilevato = True
            print("[RILEVAMENTO] Incrocio rilevato: la linea guida si avvicina molto o si interrompe.")

        # Disegno indicatori grafici
        cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
        cv2.line(frame, (centro_camera, altezza // 2), (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)
    else: 
        # Se la linea sparisce del tutto proprio mentre avanziamo verso l'apertura, l'abbiamo persa entrando nell'incrocio
        errore_x = None  
        dati_condivisi["frame_linea_persa"] += 1
        if dati_condivisi["frame_linea_persa"] > SOGLIA_ALLARME_LINEA_PERSA:
            incrocio_rilevato = True
        else: 
            incrocio_rilevato = False
        
    return frame, mask, incrocio_rilevato, errore_x


# ==========================================
# 7. MACCHINA A STATI LOGICA DECISIONALE
# ==========================================
def gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili):
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta,mappa_cartelli_global
    
    # ----------------------------------------------------
    # STATO: NAVIGAZIONE (Istruzioni di centraggio sulla linea)
    # ----------------------------------------------------
    if stato_attuale == STATO_NAVIGAZIONE:
        if incrocio_rilevato:
            stato_attuale = STATO_INCROCIO_ATTESA
            mappa_cartelli_global = {}          # Reset preventivo mappa globale
            dati_condivisi["ocr_cartelli"] = {}  # Reset preventivo cartelli OCR
            invia_voce("Incrocio rilevato. Ti prego di fermarti. Sto leggendo i cartelli.", prioritario=True)

        else:
            if errore_x is not None:
                tolleranza = 55
                if errore_x > tolleranza:
                    invia_voce("Fai un piccolo passo laterale verso destra.")
                elif errore_x < -tolleranza:
                    invia_voce("Fai un piccolo passo laterale verso sinistra.")
                else:
                    invia_voce("Sei centrato sulla linea. Procedi dritto.")
            else:
                invia_voce("Attenzione, percorso non rilevato. Cammina lentamente cercando la linea blu.")

    # ----------------------------------------------------
    # STATO: ATTESA ALL'INCROCIO (Lettura cartelli e scelta direzione)
    # ----------------------------------------------------
    elif stato_attuale == STATO_INCROCIO_ATTESA:
        # Appena l'OCR riempie la coda e la mappa globale è ancora vuota (non annunciata)
        if cartelli_disponibili and not mappa_cartelli_global:
            # Sincronizza la mappa globale per consentire il funzionamento del Microfono
            mappa_cartelli_global = cartelli_disponibili.copy()

            # Annuncia vocalmente i cartelli trovati
            elenco = ", oppure ".join(list(cartelli_disponibili.keys()))
            invia_voce(f" Ho Letto : {elenco}.", prioritario=True)

            # --- LOGICA DI TRANSIZIONE ALLO STATO DI SVOLTA AUTOMATICA ---
            # Estraiamo la prima direzione valida trovata nei cartelli
            for testo, direzione in cartelli_disponibili.items():
                if direzione in ["DESTRA", "SINISTRA", "DRITTO"]:
                    direzione_da_prendere = direzione
                    tempo_inizio_svolta = time.time()        # Salva il timestamp di inizio manovra
                    stato_attuale = STATO_SVOLTA_AUTOMATICA   
                    print(f"[STATO] Transizione a SVOLTA AUTOMATICA. Direzione: {direzione_da_prendere}")

                    
                    if direzione == "SINISTRA":
                        invia_voce(f"Ho letto {testo}. Ruota adesso a sinistra sul posto di novanta gradi.", prioritario=True)
                    elif direzione == "DESTRA":
                        invia_voce(f"Ho letto {testo}. Ruota adesso a destra sul posto di novanta gradi.", prioritario=True)
                    else: # DRITTO
                        invia_voce(f"Ho letto {testo}. Procedi dritto in avanti per superare l'incrocio.", prioritario=True)
                    
                    print(f"[STATO] Transizione a SVOLTA AUTOMATICA. Direzione: {direzione_da_prendere}")
                    break # Usciamo dal ciclo dei cartelli


    # ----------------------------------------------------
    # STATO: SVOLTA GUIDATA (Accompagnamento acustico di manovra)
    # --------------------------------------------
    # 
    # --------
    elif stato_attuale == STATO_SVOLTA_AUTOMATICA:
        
        if direzione_da_prendere in ["DESTRA", "SINISTRA"]:
            durata_manovra = 4.5  # Durata stimata per una rotazione di 90 gradi
        else:  # DRITTO
            durata_manovra = 2.5  # Durata stimata per attraversare l'incrocio
            
        # Finita la durata stimata della manovra a piedi, l'assistente torna a tracciare la linea
        if time.time() - tempo_inizio_svolta > durata_manovra:
            invia_voce("Manovra completata. Riprendo l'assistenza sul percorso lineare.", prioritario=True)
            dati_condivisi["ocr_cartelli"] = {}  # Pulisci i vecchi dati
            mappa_cartelli_global = {}           # Pulisci i vecchi dati
            stato_attuale = STATO_NAVIGAZIONE
            print("[STATO] Manovra conclusa. Ritorno a NAVIGAZIONE.")


# ==========================================
# 8. OVERLAY VISIVO DI MONITORAGGIO 
# ==========================================
def applica_overlay_grafico(frame, stato, mappa_cartelli):
    cv2.putText(frame, f"STATO ASSISTENTE: {stato}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    if stato == STATO_SVOLTA_AUTOMATICA:
        cv2.putText(frame, f"GUIDA ALLA MANOVRA: {direzione_da_prendere}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


# ==========================================
# 6. ORCHESTRATORE MAIN LOOP
# ==========================================
def main():
    global frame_condiviso, mappa_cartelli_global, running
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta
    
    

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
    # Inizializza la webcam del computer/telecamera esterna (ID 0 o 1)
    video = cv2.VideoCapture(1)
    
    # AVVIO DI TUTTI E 4 I THREAD DI BACKGROUND
    threading.Thread(target=thread_yolo,args=(yolo_model,) , daemon=True).start()
    threading.Thread(target=thread_elaborazione_ocr,args=(lettore_ocr,) , daemon=True).start()
    threading.Thread(target=thread_notifiche_vocali, daemon=True).start()
    
    invia_voce("Dispositivo di assistenza attivo. Puoi iniziare a camminare.", prioritario=True)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("[ERRORE] Impossibile acquisire il video dalla telecamera.")
            break


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
            
        # Carica il frame corrente nella memoria condivisa per l'OCR thread
        dati_condivisi["frame_da_analizzare"] = frame.copy()
        cartelli_disponibili = dati_condivisi["ocr_cartelli"].copy()

        
        # FASE 2: ANALISI ISOLAMENTO LINEE GUIDA BLU
        frame_linee, maschera_linee, incrocio_rilevato, errore_x = elabora_linee_guida(frame.copy(), BLU_LOWER, BLU_UPPER)

        
        
        # FASE 3: MACCHINA A STATI (Genera i comandi vocali in coda)
        gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili)
        
        # Mostriamo a schermo i dati (per chi monitora il test da PC)
        cv2.putText(frame_linee, f"STATO: {stato_attuale}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        frame_finale = applica_overlay_grafico(frame_linee, stato_attuale, cartelli_disponibili)
        cv2.imshow("Maschera", maschera_linee)
        cv2.imshow("Pepper Navigation System", frame_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Chiusura sicura dei flussi hardware
    dati_condivisi["running"] = False
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()