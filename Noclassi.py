import os
import cv2
import numpy as np
import threading
import time
import easyocr
import queue
import speech_recognition as sr
from cvlib.object_detection import YOLO
import winsound  # Riconoscimento audio e feedback acustici di sistema





# --- STRUMENTI DI SINCRONIZZAZIONE (CONCORRENZA) ---
coda_testo_da_leggere = queue.Queue() # Passa i testi dal Main/Orecchio alla Bocca

# IL SEMAFORO! (Event)
# False = Rosso (La bocca sta parlando, l'orecchio deve aspettare)
# True = Verde (Silenzio, l'orecchio può registrare)
semaforo_voce = threading.Event()
semaforo_voce.set() # All'avvio è verde


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


#Parametri per condizioni di luce variabili (es. esterno con sole, interno con ombre)
BLU_LOWER = np.array([95, 70, 30])
BLU_UPPER = np.array([135, 255, 255])

# Configurazione Soglie Visive
SOGLIA_MINIMA_PIXEL = 3000       # Sotto questa quota di pixel blu, la linea è considerata persa
SOGLIA_Y_INCROCIO = 320          # Se il baricentro blu scende troppo in basso (es. > 320 su 480 di altezza), l'incrocio è vicino

# --- PARAMETRI GESTIONE LINEA PERSA E RIALLINEAMENTO ---
SOGLIA_FRAME_LINEA_PERSA = 15    # Frame senza linea → avviso di smarrimento
SOGLIA_FRAME_PERSA_MEDIA = 45    # Frame → istruzione direzionale precisa
SOGLIA_FRAME_PERSA_TOTALE = 90   # Frame → procedura di recupero completo
ultima_direzione_errore = None   # "sinistra" o "destra": da che lato era uscita la linea
contatore_linea_persa = 0        # Conta i fotogrammi consecutivi senza linea
ultima_x_valida = None           # Memorizza l'ultima coordinata X della linea prima di perderla
fase_recupero_annunciata = 0     # Quale fase di recupero è già stata annunciata (0/1/2/3)
diramazione_vista_prima = False  # True solo se nei frame precedenti erano visibili 2+ rami: evita
                                 # falsi incrocci quando la camera si sposta o la linea sparisce

# ---------------------------------------------------------------
# PARAMETRI INTERVALLO DI CONFIDENZA - NAVIGAZIONE LINEA BLU
# ---------------------------------------------------------------
# L'errore_x misura la distanza in pixel tra il centro della linea
# e il centro della fotocamera (asse ottico). Suddividiamo in 4 zone:
#
#   |---ZONA_CRITICA---|---ZONA_CORREZIONE---|---ZONA_OK---|---ZONA_OK---|---ZONA_CORREZIONE---|---ZONA_CRITICA---|
#   0                                        centro-TOL  centro+TOL                          larghezza
#
# ZONA_OK        (|errore_x| <= TOLLERANZA_OK):    Utente centrato → "Procedi dritto"
# ZONA_LEGGERA   (TOLLERANZA_OK < |errore_x| <= TOLLERANZA_MEDIA): Lieve scostamento → passo laterale lieve
# ZONA_DECISA    (TOLLERANZA_MEDIA < |errore_x| <= TOLLERANZA_FORTE): Scostamento netto → passo deciso
# ZONA_CRITICA   (|errore_x| > TOLLERANZA_FORTE): Rischio perdita linea imminente → correzione urgente

TOLLERANZA_OK     = 60   # pixel: zona verde (centrato)
TOLLERANZA_MEDIA  = 130  # pixel: correzione lieve
TOLLERANZA_FORTE  = 210  # pixel: correzione decisa (oltre → rischio perdita)

# Storico errori per stabilizzare i comandi (evita oscillazioni)
SMOOTHING_BUFFER_SIZE = 5        # Numero di frame su cui mediare l'errore_x
_buffer_errore_x = []            # Buffer circolare interno

# STATI DELL'ASSISTENTE INDOSSABILE
STATO_NAVIGAZIONE = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"
STATO_LINEA_PERSA = "LINEA_PERSA"

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

# Flag: True mentre si attende che l'utente torni al centro dopo una correzione.
# Blocca la ripetizione del comando di correzione finché il centro non è ritrovato.
correzione_in_corso = False

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
                        direzione = None

                        # --- 1. CONTROLLO SEMANTICO (Priorità al significato del testo) ---
                        if "DESTRA" in testo_pulito:
                            direzione = "DESTRA"
                        elif "SINISTRA" in testo_pulito:
                            direzione = "SINISTRA"
                        elif "DRITTO" in testo_pulito or "AVANTI" in testo_pulito:
                            direzione = "DRITTO"

                        if direzione is not None:
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

# 1. Usa UNA SOLA CODA per tutta la voce del sistema
coda_voce_unica = queue.Queue()

# 2. Unico Thread per la gestione del Text-To-Speech
def thread_bocca_tts_unico():
    import pythoncom
    import win32com.client
    pythoncom.CoInitialize()  # Obbligatorio per usare COM in un thread secondario

    voce = win32com.client.Dispatch("SAPI.SpVoice")
    voce.Rate = 1  # Velocità: da -10 (lento) a +10 (veloce). 1 = leggermente sopra il normale

    print("[THREAD VOCE UNIFICATO] Pronto.")
    while dati_condivisi["running"]:
        try:
            testo = coda_voce_unica.get(timeout=0.5)

            semaforo_voce.clear()  # Rosso: il robot sta parlando, l'orecchio aspetta
            print(f"\n>>> [SISTEMA PARLA]: {testo} <<<\n")

            voce.Speak(testo)      # Sincrono per default: blocca finché non ha finito

            semaforo_voce.set()    # Verde: il robot ha finito, l'orecchio può ascoltare
            coda_voce_unica.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERRORE AUDIO] {e}")
            semaforo_voce.set()    # In caso di errore, sblocca comunque il semaforo

# 3. Aggiorna la funzione invia_voce per usare la nuova coda
def invia_voce(testo, prioritario=False):
    global ultimo_messaggio_navigazione, ultimo_tempo_voce
    tempo_attuale = time.time()

    if prioritario:
        semaforo_voce.clear()  # ← Solo qui, quando siamo CERTI di parlare
        while not coda_voce_unica.empty():
            try:
                coda_voce_unica.get_nowait()
                coda_voce_unica.task_done()
            except queue.Empty:
                break
        coda_voce_unica.put(testo)
        ultimo_messaggio_navigazione = ''
        ultimo_tempo_voce = 0

    else:
        if testo != ultimo_messaggio_navigazione or (tempo_attuale - ultimo_tempo_voce) > INTERVALLO_VOCE:
            semaforo_voce.clear()  # ← Solo se il messaggio viene davvero messo in coda
            coda_voce_unica.put(testo)
            ultimo_messaggio_navigazione = testo
            ultimo_tempo_voce = tempo_attuale
        # Se scartato, non toccare il semaforo → rimane verde, l'orecchio continua ad ascoltare


# ====================================
# 4. THREAD GESTIONE COMANDI TASTIERA
# ====================================

def gestione_comandi_tastiera(tasto,monitoraggio_attivo,ultimo_annuncio_ambiente):
    """
    Gestisce l'attivazione del monitoraggio con il tasto 'W' 
    e la disattivazione con il tasto 'S'.
    
    Ritorna il nuovo stato del monitoraggio (True o False).
    """
    continua_ciclo = True
    #Controllo se viene premuto 'W' (sia minuscolo che maiuscolo)
    if tasto == ord('w') or tasto == ord('W'):
        if not monitoraggio_attivo:
            monitoraggio_attivo = True
            ultimo_annuncio_ambiente = 0  # Forza la lettura vocale immediata degli oggetti
            invia_voce("Monitoraggio ambientale attivato. Metti il telefono parallelo al petto.", prioritario=True)
            print("[SISTEMA] Monitoraggio Ambientale Continuo: ACCESO")
            
    # Controllo se viene premuto 'S' (sia minuscolo che maiuscolo)
    elif tasto == ord('s') or tasto == ord('S'):
        if monitoraggio_attivo:
            monitoraggio_attivo = False
            invia_voce("Monitoraggio ambientale disattivato. Metti il telefono parallelo al pavimento. Torno alla navigazione.", prioritario=True)
            print("[SISTEMA] Monitoraggio Ambientale Continuo: SPENTO")
            
    return monitoraggio_attivo,ultimo_annuncio_ambiente,continua_ciclo




# ==========================================
# 6. FUNZIONE ELABORAZIONE LINEE GUIDA BLU
# ==========================================
def elabora_linee_guida(frame, blu_lower, blu_upper):
    global contatore_linea_persa, ultima_x_valida, ultima_direzione_errore
    global diramazione_vista_prima, _buffer_errore_x
    altezza, larghezza, _ = frame.shape
    centro_camera = larghezza // 2
    incrocio_rilevato = False
    errore_x = 0
    
    # Segmentazione colore HSV e pulizia
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blu_lower, blu_upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # TROVA TUTTI I CONTORNI (i blocchi bianchi separati)
    contorni, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contorni_validi = []
    for c in contorni:
        if cv2.contourArea(c) > 1500: 
            contorni_validi.append(c)
            
    # --- LOGICA DI DIRAMAZIONE (incrocio con linea visibile) ---
    if len(contorni_validi) >= 2:
        aree = sorted([cv2.contourArea(c) for c in contorni_validi], reverse=True)
        area_principale = aree[0]
        area_secondaria = aree[1]
        if area_secondaria >= area_principale * 0.25:
            incrocio_rilevato = True
            diramazione_vista_prima = True
        
    # Calcoliamo il centraggio sul contorno più grande (la linea principale)
    if len(contorni_validi) > 0:
        contorno_maggiore = max(contorni_validi, key=cv2.contourArea)
        moments = cv2.moments(contorno_maggiore)
        
        if moments["m00"] > 0:
            centro_linea_guida_x = int(moments["m10"] / moments["m00"])
            centro_linea_guida_y = int(moments["m01"] / moments["m00"])

            contatore_linea_persa = 0
            ultima_x_valida = centro_linea_guida_x
            errore_x_raw = centro_linea_guida_x - centro_camera

            # --- SMOOTHING: media mobile sull'errore per stabilizzare i comandi ---
            _buffer_errore_x.append(errore_x_raw)
            if len(_buffer_errore_x) > SMOOTHING_BUFFER_SIZE:
                _buffer_errore_x.pop(0)
            errore_x = int(np.mean(_buffer_errore_x))

            # Aggiorna la direzione di scostamento con isteresi (aggiorna solo se significativo)
            if errore_x > TOLLERANZA_OK:
                ultima_direzione_errore = "destra"
            elif errore_x < -TOLLERANZA_OK:
                ultima_direzione_errore = "sinistra"

            # --- VISUALIZZAZIONE CONFIDENZA ---
            # Linea verticale: asse ottico camera (centro frame)
            cv2.line(frame, (centro_camera, 0), (centro_camera, altezza), (255, 255, 0), 1)
            # Linea orizzontale: baricentro Y della linea
            cv2.line(frame, (0, centro_linea_guida_y), (larghezza, centro_linea_guida_y), (255, 255, 0), 1)
            # Punto baricentro linea guida
            cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
            # Linea che collega baricentro linea → centro camera (asse di errore)
            cv2.line(frame, (centro_linea_guida_x, centro_linea_guida_y),
                     (centro_camera, centro_linea_guida_y), (0, 140, 255), 2)
            # Zona verde OK
            cv2.rectangle(frame,
                          (centro_camera - TOLLERANZA_OK, 0),
                          (centro_camera + TOLLERANZA_OK, altezza),
                          (0, 180, 0), 1)
            # Zona gialla (correzione lieve)
            cv2.rectangle(frame,
                          (centro_camera - TOLLERANZA_MEDIA, 0),
                          (centro_camera + TOLLERANZA_MEDIA, altezza),
                          (0, 220, 220), 1)
            # Zona rossa (correzione forte / rischio perdita)
            cv2.rectangle(frame,
                          (centro_camera - TOLLERANZA_FORTE, 0),
                          (centro_camera + TOLLERANZA_FORTE, altezza),
                          (0, 0, 200), 1)
            # Testo distanza in pixel
            cv2.putText(frame, f"Errore: {errore_x:+d}px", (10, altezza - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

            cv2.drawContours(frame, contorni_validi, -1, (0, 255, 0), 2)
    else:
        errore_x = None
        _buffer_errore_x.clear()   # Reset buffer quando la linea è persa
        contatore_linea_persa += 1

        # --- LOGICA LINEA PERSA → incrocio solo se stavamo davvero in un bivio ---
        if (contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA
                and ultima_x_valida is not None
                and diramazione_vista_prima):
            incrocio_rilevato = True
        else:
            incrocio_rilevato = False
            if contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA:
                diramazione_vista_prima = False

        
    # Scriviamo sullo schermo quante linee vede
    cv2.putText(frame, f"Linee rilevate: {len(contorni_validi)}", (10, altezza - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
    return frame, mask, incrocio_rilevato, errore_x

# ==========================================
# 7. MACCHINA A STATI LOGICA DECISIONALE
# ==========================================
def gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili,larghezza):
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta, mappa_cartelli_global
    global contatore_linea_persa, fase_recupero_annunciata, diramazione_vista_prima
    global correzione_in_corso

    centro_camera = larghezza // 2
    
    # ----------------------------------------------------
    # STATO: NAVIGAZIONE (Istruzioni di centraggio sulla linea)
    # ----------------------------------------------------
    if stato_attuale == STATO_NAVIGAZIONE:
        if incrocio_rilevato:
            correzione_in_corso = False   # Reset flag al cambio di stato
            stato_attuale = STATO_INCROCIO_ATTESA
            mappa_cartelli_global = {}
            dati_condivisi["ocr_cartelli"] = {}
            invia_voce("Incrocio rilevato. Rallenta e fermati. Sto leggendo i cartelli.", prioritario=True)

        elif errore_x is not None:
            # LINEA VISIBILE - Comandi proporzionali all'errore (4 zone di confidenza)
            fase_recupero_annunciata = 0
            abs_err = abs(errore_x)
            lato = "destra" if errore_x < 0 else "sinistra"

            if abs_err <= TOLLERANZA_OK:
                # CENTRO RITROVATO: resetta il blocco e conferma solo se si era in correzione
                if correzione_in_corso:
                    correzione_in_corso = False
                    invia_voce("Centro ritrovato. Procedi dritto.", prioritario=True)
                else:
                    invia_voce("Sei centrato sulla linea. Procedi dritto.")

            else:
                # FUORI DAL CENTRO: avvisa solo se non è già in corso una manovra di correzione
                if not correzione_in_corso:
                    correzione_in_corso = True
                    if lato == "destra":
                        invia_voce(f"Attenzione! Stai uscendo dalla linea verso {lato}. Correzione immediata verso sinistra.", prioritario=True)
                    else:
                        invia_voce(f"Attenzione! Stai uscendo dalla linea verso {lato}. Correzione immediata verso destra.", prioritario=True)
                # Se correzione_in_corso è già True: silenzio, si aspetta il ritrovamento del centro

        else:
            # --- LOGICA DI GESTIONE E RI-ALLINEAMENTO LINEA PERSA ---
            if contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA:
                correzione_in_corso = False   # Reset flag al cambio di stato
                stato_attuale = STATO_LINEA_PERSA
                fase_recupero_annunciata = 0
                print("[STATO] Transizione a LINEA_PERSA.")


# ----------------------------------------------------
# STATO: LINEA PERSA (Recupero progressivo in 3 fasi)
# ----------------------------------------------------
    elif stato_attuale == STATO_LINEA_PERSA:

        # Se la linea ricompare, torniamo subito in NAVIGAZIONE
        if errore_x is not None:
            fase_recupero_annunciata = 0
            stato_attuale = STATO_NAVIGAZIONE
            invia_voce("Linea ritrovata. Riprendo la navigazione.", prioritario=True)
            return

        # ---------------------------------------------------------------
        # LOGICA DI RECUPERO DIREZIONE
        # ---------------------------------------------------------------
        # ultima_direzione_errore indica verso dove si trovava la linea
        # rispetto al centro camera PRIMA di essere persa.
        #
        # Esempio: se l'errore era positivo → la linea era a DESTRA del centro
        # → significa che l'utente si è spostato verso SINISTRA rispetto alla linea
        # → per recuperarla, l'utente deve ruotare/spostarsi verso DESTRA.
        #
        # La dir_recupero è quindi uguale a ultima_direzione_errore:
        # la linea era a destra → vai a destra per ritrovarla.
        # ---------------------------------------------------------------
        if ultima_direzione_errore == "destra":
            dir_recupero = "destra"
            dir_opposta  = "sinistra"
        elif ultima_direzione_errore == "sinistra":
            dir_recupero = "sinistra"
            dir_opposta  = "destra"
        else:
            dir_recupero = None
            dir_opposta  = None

        # --- FASE 1: avviso immediato (soglia minima superata) ---
        if contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA and fase_recupero_annunciata < 1:
            fase_recupero_annunciata = 1
            if dir_recupero:
                invia_voce(
                    f"Attenzione, linea persa. L'ultima posizione era a {dir_recupero} del tuo campo visivo. "
                    f"Fermati e ruota lentamente verso {dir_recupero}.",
                    prioritario=True
                )
            else:
                invia_voce("Attenzione, linea persa. Fermati e ruota lentamente cercando la linea.", prioritario=True)
            print(f"[RECUPERO FASE 1] contatore={contatore_linea_persa}")

        # --- FASE 2: istruzione angolare precisa ---
        elif contatore_linea_persa >= SOGLIA_FRAME_PERSA_MEDIA and fase_recupero_annunciata < 2:
            fase_recupero_annunciata = 2
            if dir_recupero:
                invia_voce(
                    f"Linea ancora assente. Ruota di trenta gradi verso {dir_recupero} "
                    f"e avanza di un passo.",
                    prioritario=True
                )
            else:
                invia_voce("Linea ancora assente. Ruota di trenta gradi e avanza di un passo.", prioritario=True)
            print(f"[RECUPERO FASE 2] contatore={contatore_linea_persa}")

        # --- FASE 3: recupero totale, scansione a ventaglio ---
        elif contatore_linea_persa >= SOGLIA_FRAME_PERSA_TOTALE and fase_recupero_annunciata < 3:
            fase_recupero_annunciata = 3
            if dir_recupero:
                invia_voce(
                    f"Percorso completamente smarrito. Torna indietro di due passi, "
                    f"poi ruota verso {dir_recupero} e cerca la linea passando lentamente "
                    f"da {dir_opposta} verso {dir_recupero}.",
                    prioritario=True
                )
            else:
                invia_voce(
                    "Percorso completamente smarrito. Torna indietro di due passi "
                    "e ruota lentamente su te stesso per cercare la linea.",
                    prioritario=True
                )
            print(f"[RECUPERO FASE 3] contatore={contatore_linea_persa}")


    # ----------------------------------------------------
    # STATO: ATTESA ALL'INCROCIO (Lettura cartelli e scelta direzione)
    # ----------------------------------------------------
    elif stato_attuale == STATO_INCROCIO_ATTESA:

        if not incrocio_rilevato and errore_x is not None:
            stato_attuale = STATO_NAVIGAZIONE
            invia_voce("Linea ripresa. Continuo la navigazione.", prioritario=True)
            return

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
            diramazione_vista_prima = False      # Reset: il prossimo incrocio va rilevato da zero
            stato_attuale = STATO_NAVIGAZIONE
            print("[STATO] Manovra conclusa. Ritorno a NAVIGAZIONE.")


# ==========================================
# 8. OVERLAY VISIVO DI MONITORAGGIO 
# ==========================================
def applica_overlay_grafico(frame, stato, mappa_cartelli):
    # Se il sistema è in modalità ascolto vocale, mostralo chiaramente
    if monitoraggio_ambientale_attivo:
        cv2.putText(frame, "STATO ASSISTENTE: ASCOLTO VOCALE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    else:
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
    global monitoraggio_ambientale_attivo
    
    # --- 1. NUOVE VARIABILI DI STATO PER IL MONITORAGGIO ---
    monitoraggio_ambientale_attivo = False
    ultimo_annuncio_ambiente = 0

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
    threading.Thread(target=thread_bocca_tts_unico, daemon=True).start()
    
    invia_voce("Dispositivo di assistenza attivo e in ascolto. Puoi iniziare a camminare.", prioritario=True)
    
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

        
        #Logica Attivazione YOLO
        if monitoraggio_ambientale_attivo:
            tempo_corrente = time.time()
            if tempo_corrente - ultimo_annuncio_ambiente > 7.0:
                # Estraiamo i dati correnti in modo sicuro senza piantarci in loop stabili
                labels_viste = list(dati_condivisi.get("yolo_labels", []))
                bbox_visti   = list(dati_condivisi.get("yolo_bbox", []))

                if not labels_viste:
                    # Se non c'è nulla o YOLO sta ancora calcolando il primissimo frame, non blocchiamo il programma
                    print("[AMBIENTE] Nessun oggetto rilevato nei thread di background...")
                else:
                    oggetti_descritti = []
                    TRADUZIONI = {
                        'person': 'una persona', 'dog': 'un cane', 'cat': 'un gatto',
                        'chair': 'una sedia', 'sofa': 'un divano', 'bed': 'un letto',
                        'dining table': 'un tavolo', 'toilet': 'un bagno',
                        'refrigerator': 'un frigorifero', 'tv': 'una televisione',
                        'laptop': 'un computer portatile', 'cell phone': 'un telefono',
                        'bottle': 'una bottiglia', 'cup': 'una tazza',
                        'book': 'un libro', 'backpack': 'uno zaino',
                        'umbrella': 'un ombrello', 'suitcase': 'una valigia'
                    }
                    for i, label in enumerate(labels_viste):
                        try:
                            x_min, y_min, x_max, y_max = bbox_visti[i]
                            alt_pixel   = y_max - y_min
                            distanza_cm = calcola_distanza(ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT), alt_pixel)
                            distanza_m  = distanza_cm / 100.0
                            centro_x    = (x_min + x_max) / 2
                            larghezza_f = frame.shape[1]
                            
                            if centro_x < larghezza_f / 3:
                                lato = "alla tua sinistra"
                            elif centro_x > larghezza_f * 2 / 3:
                                lato = "alla tua destra"
                            else:
                                lato = "davanti a te"
                            nome_ita = TRADUZIONI.get(label, f"un oggetto ({label})")
                            oggetti_descritti.append(f"{nome_ita} {lato}, a circa {distanza_m:.1f} metri")
                        except (IndexError, Exception):
                            pass

                    if oggetti_descritti:
                        descrizione = "Nell'ambiente rilevo: " + "; ".join(oggetti_descritti) + "."
                    else:
                        descrizione = "Non riesco a stimare la distanza degli oggetti rilevati."
                    invia_voce(descrizione, prioritario=True)

                semaforo_voce.wait()
                ultimo_annuncio_ambiente = tempo_corrente
            

            # Comando gestito (non di chiusura): resetta stato e riattiva la navigazione
            # Svuota anche la coda TTS da eventuali messaggi accodati durante il freeze
            stato_attuale = STATO_NAVIGAZIONE
            while not coda_voce_unica.empty():
                try: coda_voce_unica.get_nowait(); coda_voce_unica.task_done()
                except: break

            
        # Carica il frame corrente nella memoria condivisa per l'OCR thread
        dati_condivisi["frame_da_analizzare"] = frame.copy()
        cartelli_disponibili = dati_condivisi["ocr_cartelli"].copy()

        
        # FASE 2: ANALISI ISOLAMENTO LINEE GUIDA BLU
        frame_linee, maschera_linee, incrocio_rilevato, errore_x = elabora_linee_guida(frame.copy(), BLU_LOWER, BLU_UPPER)
        
        
        # FASE 3: MACCHINA A STATI (Genera i comandi vocali in coda)
        # Congelata mentre il sistema è in modalità ascolto vocale attivo:
        # evita che indicazioni di percorso si sovrappongano alla gestione del comando.
        if not monitoraggio_ambientale_attivo:
            gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili, larghezza=frame.shape[1])

        
        # --- INTERCETTAZIONE TASTIERA REFACTORIZZATA ---
        tasto = cv2.waitKey(1) & 0xFF
        
        # Utilizzo della funzione esterna per l'aggiornamento dei parametri
        monitoraggio_ambientale_attivo, ultimo_annuncio_ambiente,continua = gestione_comandi_tastiera(
            tasto, monitoraggio_ambientale_attivo, ultimo_annuncio_ambiente
        )

        if not continua:
            break
        
        # Mostriamo a schermo i dati (per chi monitora il test da PC)
        cv2.putText(frame_linee, f"STATO: {stato_attuale}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        frame_finale = applica_overlay_grafico(frame_linee, stato_attuale, cartelli_disponibili)
        cv2.imshow("Maschera", maschera_linee)
        cv2.imshow("Navigation System", frame_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Chiusura sicura dei flussi hardware
    dati_condivisi["running"] = False
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()