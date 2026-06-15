import cv2
import numpy as np
import threading
import time
import easyocr
import queue
import warnings
import pyttsx3 # Assicurati che sia importato in cima al file

# Inizializzazione globale del motore vocale nel Main Thread
engine_voce = pyttsx3.init()
rate = engine_voce.getProperty('rate')
engine_voce.setProperty('rate', rate - 30)

# ==========================================
# INIZIALIZZAZIONE MODULI AI (OCR)
# ==========================================
print("[INFO] Inizializzazione motore OCR in corso...")
lettore_ocr = easyocr.Reader(['it', 'en'], gpu=False)

# Variabili condivise tra i thread
frame_condiviso = None
testo_rilevato_global = "In attesa di cartelli..."
running = True
mappa_cartelli_global = {}  # Conterrà { "TESTO_CARTELLO": "DIREZIONE" }

# Configurazione Soglie Visive calibrate sul tuo tracciato reale
SOGLIA_MINIMA_PIXEL = 2000       # Sotto questa quota di pixel blu, la linea è considerata persa
SOGLIA_Y_INCROCIO = 400          # Se il baricentro blu scende troppo in basso, l'incrocio è vicino

# STATI DELL'ASSISTENTE INDOSSABILE
STATO_NAVIGAZIONE = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"

stato_attuale = STATO_NAVIGAZIONE
direzione_da_prendere = None  
tempo_inizio_svolta = 0

# Gestione Coda Vocale Asincrona
coda_voce = queue.Queue()
ultimo_messaggio_navigazione = ""
ultimo_tempo_voce = 0
INTERVALLO_VOCE = 2.0  # Secondi di silenzio obbligatori tra indicazioni ripetitive di centraggio

# ==========================================
# 1. THREAD OUTPUT VOCALE (Sintesi Vocale attiva)
# ==========================================
def thread_notifiche_vocali():
    global running
    print("[THREAD OUTPUT VOCALE] Pronto e parlante.")
    while running:
        try:
            # Estrae il messaggio dalla coda (si blocca per massimo 0.5 secondi se vuota)
            messaggio = coda_voce.get(timeout=0.5)
            print(f"\n>>> [GUIDA VOCALE]: {messaggio} <<<\n")
            
            # Riproduzione audio reale
            engine_voce.say(messaggio)
            engine_voce.runAndWait()
            
            coda_voce.task_done()
            time.sleep(0.1)  # Piccola pausa per evitare sovrapposizioni vocali
        except queue.Empty:
            continue

def invia_voce(testo, prioritario=False):
    """Pronuncia il testo usando il motore globale del thread principale"""
    global ultimo_messaggio_navigazione, ultimo_tempo_voce
    tempo_attuale = time.time()
    
    if prioritario:
        print(f"\n>>> [GUIDA VOCALE PRIORITARIA]: {testo} <<<\n")
        
        # --- FIX SVUOTAMENTO CODA ---
        # Se il messaggio è prioritario (Incrocio), forziamo il motore a fermare 
        # immediatamente qualsiasi riproduzione residua (es. "procedi dritto")
        try:
            engine_voce.stop() 
        except Exception:
            pass
        engine_voce.say(testo)
        engine_voce.runAndWait()  # Blocca e parla subito
        ultimo_messaggio_navigazione = testo
    else:
        # Messaggi standard di navigazione
        if testo != ultimo_messaggio_navigazione or (tempo_attuale - ultimo_tempo_voce) > INTERVALLO_VOCE:
            print(f"\n>>> [GUIDA VOCALE]: {testo} <<<\n")
            engine_voce.say(testo)
            ultimo_messaggio_navigazione = testo
            ultimo_tempo_voce = tempo_attuale

# ==========================================
# 2. THREAD OCR: LETTURA ASINCRONA CARTELLI
# ==========================================
def thread_elaborazione_ocr():
    global frame_condiviso, mappa_cartelli_global, running
    print("[THREAD OCR] Avviato. Analisi cartelli attiva.")
    
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
            tre_zone = larghezza_frame // 3  # Dividiamo lo schermo in 3 fette verticali
            
            for (bbox, testo, probabilita) in risultati:
                testo_pulito = testo.strip().upper()
                
                # Consideriamo valido il testo se la confidenza è accettabile
                if probabilita > 0.4 and len(testo_pulito) > 1:
                    # Calcoliamo il centro geometrico X del cartello rilevato
                    x_centro = int((bbox[0][0] + bbox[1][0]) / 2)

                    # Determina la direzione spaziale del cartello rispetto all'incrocio
                    if x_centro < tre_zone:
                        direzione = "SINISTRA"
                    elif x_centro < 2 * tre_zone:
                        direzione = "DRITTO"
                    else:
                        direzione = "DESTRA"

                    nuovi_cartelli[testo_pulito] = direzione

            if nuovi_cartelli:
                mappa_cartelli_global = nuovi_cartelli
                print(f"[OCR] Rilevati cartelli: {mappa_cartelli_global}")

        except Exception as e:
            print(f"[ERRORE OCR] {e}")
            
        time.sleep(0.4)


# ==========================================
# 3. SEGMENTAZIONE COLORE E TRACCIAMENTO LINEE (OpenCV HSV)
# ==========================================
def elabora_linee_guida(frame, blu_lower, blu_upper):
    altezza, larghezza, _ = frame.shape
    centro_camera = larghezza // 2
    incrocio_rilevato = False
    errore_x = 0
    
    # 1. Trasformazione in HSV per la segmentazione del colore
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, blu_lower, blu_upper)
    
    # Pulizia morfologica (rimuove i riflessi bianchi sulle mattonelle lucide)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Calcolo dei momenti geometrici sulla maschera binaria
    moments = cv2.moments(mask)
    area_linea_guida = moments["m00"]

    if area_linea_guida > SOGLIA_MINIMA_PIXEL:
        centro_linea_guida_x = int(moments["m10"] / moments["m00"])
        centro_linea_guida_y = int(moments["m01"] / moments["m00"])
        
        # Errore di allineamento (distanza dal centro dello schermo)
        errore_x = centro_linea_guida_x - centro_camera

        # LOGICA INCROCIO: Se il baricentro scende troppo in basso, siamo vicini alla biforcazione aperta
        if centro_linea_guida_y > SOGLIA_Y_INCROCIO:
            incrocio_rilevato = True

        # Disegno indicatori grafici per il monitoraggio
        cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
        cv2.line(frame, (centro_camera, altezza // 2), (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)
    else: 
        # Se la linea scompare del tutto mentre avanziamo nell'apertura, siamo dentro l'incrocio
        errore_x = None  
        incrocio_rilevato = True  

    return frame, mask, incrocio_rilevato, errore_x


# ==========================================
# 4. MACCHINA A STATI LOGICA DECISIONALE
# ==========================================
def gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili):
    global stato_attuale, direzione_da_prendere, tempo_inizio_svolta
    
    # ----------------------------------------------------
    # STATO: NAVIGAZIONE STANDARD (Centraggio sulla linea)
    # ----------------------------------------------------
    if stato_attuale == STATO_NAVIGAZIONE:
        if incrocio_rilevato:
            stato_attuale = STATO_INCROCIO_ATTESA
            invia_voce("Incrocio rilevato. Rallenta e fermati. Sto leggendo la segnaletica.", prioritario=True)
        else:
            if errore_x is not None:
                tolleranza = 50
                if errore_x > tolleranza:
                    invia_voce("Fai un piccolo passo a destra.")
                elif errore_x < -tolleranza:
                    invia_voce("Fai un piccolo passo a sinistra.")
                else:
                    invia_voce("Procedi dritto sulla linea.")
            else:
                invia_voce("Attenzione, linea non rilevata. Cammina lentamente cercandola.")

    # ----------------------------------------------------
    # STATO: ATTESA DECISIONALE AUTOMATICA (Analisi Cartelli letti dall'OCR)
    # ----------------------------------------------------
    elif stato_attuale == STATO_INCROCIO_ATTESA:
        if cartelli_disponibili:
            # Algoritmo decisionale: analizziamo i testi letti
            for testo_cartello, direzione_associata in cartelli_disponibili.items():
                print(f"[DECISIONE] Analizzo testo letto: '{testo_cartello}' -> va a {direzione_associata}")
                
                # Esempio di comportamento intelligente basato sul testo del cartello
                # Se il testo contiene indicazioni direzionali o parole chiave del tuo progetto
                if "DESTRA" in testo_cartello or "RIGHT" in testo_cartello:
                    direzione_da_prendere = "DESTRA"
                elif "SINISTRA" in testo_cartello or "LEFT" in testo_cartello:
                    direzione_da_prendere = "SINISTRA"
                else:
                    # Se legge una destinazione generica (es. "MENSE"), assegna la direzione spaziale in terzi calcolata dal thread OCR
                    direzione_da_prendere = direzione_associata
                
                # Comunichiamo la decisione presa e cambiamo stato
                invia_voce(f"Rilevato cartello: {testo_cartello}. Indicazione per svolta a {direzione_da_prendere}.", prioritario=True)
                tempo_inizio_svolta = time.time()
                stato_attuale = STATO_SVOLTA_AUTOMATICA
                break
        else:
            # Resta fermo in attesa che l'OCR veda qualcosa nell'inquadratura
            pass

    # ----------------------------------------------------
    # STATO: GUIDA ALLA MANOVRA VOCALE
    # ----------------------------------------------------
    elif stato_attuale == STATO_SVOLTA_AUTOMATICA:
        if direzione_da_prendere == "SINISTRA":
            invia_voce("Ruota adesso a sinistra sul posto di novanta gradi.")
            durata_manovra = 3.5
        elif direzione_da_prendere == "DESTRA":
            invia_voce("Ruota adesso a destra sul posto di novanta gradi.")
            durata_manovra = 3.5
        else: 
            invia_voce("Procedi in avanti dritto per superare l'incrocio.")
            durata_manovra = 2.0
            
        # Controllo tempo stimato per l'esecuzione della manovra a piedi
        if time.time() - tempo_inizio_svolta > durata_manovra:
            invia_voce("Manovra eseguita. Riprendo la navigazione lineare sulla linea blu.", prioritario=True)
            stato_attuale = STATO_NAVIGAZIONE


# ==========================================
# 5. OVERLAY GRAFICO DI MONITORAGGIO
# ==========================================
def applica_overlay_ocr(frame, stato, mappa_cartelli):
    cv2.putText(frame, f"STATO: {stato}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    if stato == STATO_INCROCIO_ATTESA:
        cv2.putText(frame, "ANALISI CARTELLI IN CORSO...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos = 90
        for testo, dir_associata in mappa_cartelli.items():
            cv2.putText(frame, f"[LETTO]: {testo} -> {dir_associata}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += 25
    elif stato == STATO_SVOLTA_AUTOMATICA:
        cv2.putText(frame, f"MANOVRA: {direzione_da_prendere}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


# ==========================================
# 6. ORCHESTRATORE MAIN LOOP
# ==========================================
def main():
    global frame_condiviso, mappa_cartelli_global, running, stato_attuale
    
    video = cv2.VideoCapture(1)
    
    # Range colore HSV del tuo nastro blu reale sul pavimento
    BLU_LOWER = np.array([95, 160, 60])
    BLU_UPPER = np.array([135, 255, 255])
    
    # Avvio dei thread di background per OCR e Sintesi Vocale
    threading.Thread(target=thread_elaborazione_ocr, daemon=True).start()
    #threading.Thread(target=thread_notifiche_vocali, daemon=True).start()
    
    invia_voce("Sistema di assistenza attivo. Puoi iniziare a camminare sulla linea blu.", prioritario=True)
    
    try:
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                print("[ERRORE] Impossibile leggere il flusso video.")
                break
                
            frame_condiviso = frame.copy()
            cartelli_disponibili = mappa_cartelli_global.copy()
            
            # 1. Segmentazione colore HSV e tracciamento
            frame_linee, maschera_linee, incrocio_rilevato, errore_x = elabora_linee_guida(frame.copy(), BLU_LOWER, BLU_UPPER)
            
            # 2. Controllo logico della macchina a stati
            gestisci_macchina_a_stati(errore_x, incrocio_rilevato, cartelli_disponibili)

            
            # 3. Interfaccia grafica
            frame_finale = applica_overlay_ocr(frame_linee, stato_attuale, cartelli_disponibili)
            cv2.imshow("Maschera (HSV Segmentazione)", maschera_linee)
            cv2.imshow("Pepper Navigation System", frame_finale)
            
            # Refresh delle maschere video e controllo uscita manuale con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interruzione manuale ricevuta.")
    finally:
        # Chiusura pulita delle risorse hardware
        running = False
        video.release()
        cv2.destroyAllWindows()
        print("[INFO] Risorse rilasciate con successo.")

if __name__ == "__main__":
    main()