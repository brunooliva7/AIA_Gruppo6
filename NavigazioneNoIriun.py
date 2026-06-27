import os
import cv2
import numpy as np
import threading
import time
import easyocr
import queue
import requests  # <-- Aggiunto per le richieste HTTP
import speech_recognition as sr
from cvlib.object_detection import YOLO
import winsound  # Riconoscimento audio e feedback acustici di sistema


# =============================================================================
# CONFIG — Costanti globali di calibrazione e parametri operativi del sistema.
# =============================================================================

FOCAL_LENGTH      = 300
DISTANZA_ALLARME  = 150
ALTEZZA_DEFAULT   = 50

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

TRADUZIONI_ITA = {
    'person': 'una persona', 'dog': 'un cane', 'cat': 'un gatto',
    'chair': 'una sedia', 'sofa': 'un divano', 'bed': 'un letto',
    'dining table': 'un tavolo', 'toilet': 'un bagno',
    'refrigerator': 'un frigorifero', 'tv': 'una televisione',
    'laptop': 'un computer portatile', 'cell phone': 'un telefono',
    'bottle': 'una bottiglia', 'cup': 'una tazza',
    'book': 'un libro', 'backpack': 'uno zaino',
    'umbrella': 'un ombrello', 'suitcase': 'una valigia'
}

BLU_LOWER = np.array([95, 70, 30])
BLU_UPPER = np.array([135, 255, 255])

# Filtri geometrici per il riconoscimento linea
LINEA_ASPECT_RATIO_MIN = 1.5    
LINEA_SOLIDITA_MIN     = 0.25   
LINEA_SATURAZIONE_MIN  = 60     
LINEA_Y_MIN_FRAZIONE   = 0.20   

SOGLIA_MINIMA_PIXEL       = 3000
SOGLIA_Y_INCROCIO         = 320
SOGLIA_FRAME_LINEA_PERSA  = 15
SOGLIA_FRAME_PERSA_MEDIA  = 45
SOGLIA_FRAME_PERSA_TOTALE = 90

TOLLERANZA_OK    = 60
TOLLERANZA_MEDIA = 130
TOLLERANZA_FORTE = 210
SMOOTHING_BUFFER_SIZE = 5

INTERVALLO_VOCE = 2.2

STATO_NAVIGAZIONE       = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA   = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"
STATO_LINEA_PERSA       = "LINEA_PERSA"


# =============================================================================
# MotoreVocale — Gestisce la sintesi vocale TTS inviando i messaggi via HTTP
# al server web locale (SitoWebDialogoVoice.py), che li riproduce sul telefono.
# =============================================================================

class MotoreVocale:

    TTS_URL     = "http://127.0.0.1:5000/api/invia-messaggio"
    POLLING_URL = "http://127.0.0.1:5000/api/leggi-comando"

    def __init__(self, dati_condivisi):
        self._dati              = dati_condivisi
        self._coda              = queue.Queue()
        self._semaforo          = threading.Event()
        self._semaforo.set()
        self._interruzione_voce = threading.Event()   
        self._ultimo_msg        = ""
        self._ultimo_tempo      = 0.0
        self._comando_remoto    = None                

    def avvia(self):
        threading.Thread(target=self._loop_tts,     daemon=True).start()
        threading.Thread(target=self._loop_polling,  daemon=True).start()

    def _loop_tts(self):
        print("[THREAD VOCE UNIFICATO] Pronto (invia i messaggi al server web).")
        while self._dati["running"]:
            try:
                elemento = self._coda.get(timeout=0.5)
                if isinstance(elemento, tuple):
                    testo, taglia_audio = elemento
                else:
                    testo, taglia_audio = elemento, False

                self._semaforo.clear()   
                print(f"\n>>> [INVIATO AL TELEFONO]: {testo} <<<\n")

                try:
                    requests.post(
                        self.TTS_URL,
                        json={"testo": testo, "taglia_audio": taglia_audio},
                        timeout=2
                    )
                    tempo_lettura = max(1.0, len(testo) / 10.0)
                    self._interruzione_voce.wait(timeout=tempo_lettura)
                    self._interruzione_voce.clear()
                except Exception as e:
                    print(f"[ERRORE] Invio messaggio al web fallito: {e}")

                self._semaforo.set()     
                self._coda.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERRORE AUDIO] {e}")
                self._semaforo.set()

    def _loop_polling(self):
        print("[THREAD POLLING] In ascolto di comandi dall'interfaccia web...")
        while self._dati["running"]:
            try:
                r = requests.get(self.POLLING_URL, timeout=1)
                if r.status_code == 200:
                    cmd = r.json().get("comando")
                    if cmd:
                        self._comando_remoto = cmd
            except Exception:
                pass
            time.sleep(0.05)

    def parla(self, testo, prioritario=False, interrompi_subito=False):
        now = time.time()

        if interrompi_subito:
            self._interruzione_voce.set()          
            self._svuota_coda_interna()
            self._coda.put((testo, True))          
            self._ultimo_msg   = ""
            self._ultimo_tempo = 0.0

        elif prioritario:
            self._svuota_coda_interna()
            self._semaforo.clear()
            self._coda.put((testo, False))
            self._ultimo_msg   = ""
            self._ultimo_tempo = 0.0

        else:
            if testo != self._ultimo_msg or (now - self._ultimo_tempo) > INTERVALLO_VOCE:
                self._semaforo.clear()

                if self._coda.qsize() > 2:
                    self._svuota_coda_interna()
                    self._coda.put((testo, True))   
                else:
                    self._coda.put((testo, False))

                self._ultimo_msg   = testo
                self._ultimo_tempo = now

    def svuota_coda(self):
        self._svuota_coda_interna()

    def _svuota_coda_interna(self):
        while not self._coda.empty():
            try:
                self._coda.get_nowait()
                self._coda.task_done()
            except queue.Empty:
                break
        self._semaforo.set()

    @property
    def semaforo(self):
        return self._semaforo

    @property
    def comando_remoto(self):
        cmd = self._comando_remoto
        self._comando_remoto = None
        return cmd


# =============================================================================
# RilevatorYOLO — Esegue il rilevamento ostacoli in un thread dedicato.
# =============================================================================

# =============================================================================
# RilevatorYOLO — Esegue il rilevamento ostacoli in un thread dedicato.
# Include la doppia stima della distanza (Focale + Ground Plane)
# =============================================================================

class RilevatorYOLO:

    def __init__(self, dati_condivisi, yolo_model):
        self._dati  = dati_condivisi
        self._model = yolo_model
        
        # ALTEZZA A CUI È POSIZIONATA LA TELECAMERA SULL'UTENTE (Modifica se serve)
        self.ALTEZZA_TELEFONO_CM = 130 

    def avvia(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        print("[THREAD] YOLO (Ostacoli) avviato.")
        while self._dati["running"]:
            if self._dati["frame_da_analizzare"] is not None and not self._dati["yolo_occupato"]:
                self._dati["yolo_occupato"] = True
                frame = self._dati["frame_da_analizzare"].copy()
                bbox, labels, conf = self._model.detect_objects(frame)
                self._dati["yolo_bbox"]     = bbox
                self._dati["yolo_labels"]   = labels
                self._dati["yolo_conf"]     = conf
                self._dati["yolo_occupato"] = False
            time.sleep(0.01)

    @staticmethod
    def calcola_distanza_altezza(altezza_reale_cm, altezza_pixel):
        """Metodo 1: Basato sull'altezza ipotetica dell'oggetto (Vostro metodo originale)"""
        if altezza_pixel <= 0: return float('inf')
        return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

    def calcola_distanza_suolo(self, y_max, altezza_frame):
        """Metodo 2: Basato sul punto in cui l'oggetto tocca il pavimento (Ground Plane)"""
        centro_y = altezza_frame / 2
        # Se la base dell'oggetto è nella metà superiore dello schermo, è all'orizzonte o sospeso
        if y_max <= centro_y:
            return float('inf')
        
        # Formula della prospettiva
        return (self.ALTEZZA_TELEFONO_CM * FOCAL_LENGTH) / (y_max - centro_y)

    def stima_distanza_migliore(self, label, y_min, y_max, altezza_frame):
        """Combina i due metodi per avere sempre la distanza in cm più sicura"""
        alt_pixel = y_max - y_min
        
        # Calcola entrambe le distanze
        dist_altezza = self.calcola_distanza_altezza(ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT), alt_pixel)
        dist_suolo = self.calcola_distanza_suolo(y_max, altezza_frame)
        
        # Restituisce la distanza MINORE (se un metodo sbaglia sovrastimando, l'altro ci salva)
        return min(dist_altezza, dist_suolo)

    def disegna_su_frame(self, frame):
        ostacolo_vicino = False
        altezza_frame = frame.shape[0]

        for i, label in enumerate(self._dati["yolo_labels"]):
            try:
                x_min, y_min, x_max, y_max = self._dati["yolo_bbox"][i]
                
                # Calcola la distanza reale ottimizzata
                distanza_cm = self.stima_distanza_migliore(label, y_min, y_max, altezza_frame)
                
                cat = "PERSONA" if label == 'person' else "OGGETTO"
                if distanza_cm < DISTANZA_ALLARME:
                    ostacolo_vicino = True
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                    cv2.putText(frame, f"ALLARME {cat} ({distanza_cm/100:.1f}m)",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    if distanza_cm != float('inf'):
                        cv2.putText(frame, f"{cat} ({distanza_cm/100:.1f}m)",
                                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except IndexError:
                pass
        return frame, ostacolo_vicino

    def descrivi_ambiente(self, frame):
        """Questa funzione genera la stringa che verrà letta vocalmente dal telefono"""
        labels_viste = list(self._dati.get("yolo_labels", []))
        bbox_visti   = list(self._dati.get("yolo_bbox", []))
        altezza_frame = frame.shape[0]

        if not labels_viste:
            print("[AMBIENTE] Nessun oggetto rilevato nei thread di background...")
            return None
            
        oggetti_descritti = []
        for i, label in enumerate(labels_viste):
            try:
                x_min, y_min, x_max, y_max = bbox_visti[i]
                
                # USA LA DISTANZA OTTIMIZZATA ANCHE PER LA VOCE
                distanza_cm = self.stima_distanza_migliore(label, y_min, y_max, altezza_frame)
                distanza_m  = distanza_cm / 100.0
                
                centro_x    = (x_min + x_max) / 2
                larghezza_f = frame.shape[1]
                
                if centro_x < larghezza_f / 3:
                    lato = "alla tua sinistra"
                elif centro_x > larghezza_f * 2 / 3:
                    lato = "alla tua destra"
                else:
                    lato = "davanti a te"
                    
                nome_ita = TRADUZIONI_ITA.get(label, f"un oggetto ({label})")
                
                # Costruisce la frase da leggere ad alta voce
                if distanza_cm != float('inf'):
                    oggetti_descritti.append(f"{nome_ita} {lato}, a circa {distanza_m:.1f} metri")
                else:
                    oggetti_descritti.append(f"{nome_ita} {lato} in lontananza")
                    
            except (IndexError, Exception):
                pass
                
        if oggetti_descritti:
            return "Nell'ambiente rilevo: " + "; ".join(oggetti_descritti) + "."
        return "Non riesco a stimare la distanza degli oggetti rilevati."


# =============================================================================
# RilevatorOCR — Legge testi da cartelli in un thread dedicato.
# =============================================================================

class RilevatorOCR:

    def __init__(self, dati_condivisi, lettore_ocr):
        self._dati    = dati_condivisi
        self._lettore = lettore_ocr

    def avvia(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        print("[THREAD OCR] Analisi cartelli attiva.")
        while self._dati["running"]:
            if self._dati["frame_da_analizzare"] is not None and not self._dati["ocr_occupato"]:
                self._dati["ocr_occupato"] = True
                img_ocr         = self._dati["frame_da_analizzare"].copy()
                gray = cv2.cvtColor(img_ocr, cv2.COLOR_BGR2GRAY)
                try:
                    risultati      = self._lettore.readtext(gray)
                    nuovi_cartelli = {}
                    for (bbox, testo, probabilita) in risultati:
                        testo_pulito = testo.strip().upper()
                        if probabilita > 0.4 and len(testo_pulito) > 1:
                            direzione = None
                            if "DESTRA" in testo_pulito:
                                direzione = "DESTRA"
                            elif "SINISTRA" in testo_pulito:
                                direzione = "SINISTRA"
                            elif "DRITTO" in testo_pulito or "AVANTI" in testo_pulito:
                                direzione = "DRITTO"
                            if direzione is not None:
                                nuovi_cartelli[testo_pulito] = direzione
                    if nuovi_cartelli:
                        self._dati["ocr_cartelli"] = nuovi_cartelli
                except Exception as e:
                    print(f"[ERRORE OCR] {e}")
                self._dati["ocr_occupato"] = False
            time.sleep(0.3)


# =============================================================================
# AnalizzatoreLinea — Elabora ogni frame per rilevare la linea blu centrale.
# =============================================================================

class AnalizzatoreLinea:

    def __init__(self):
        self._contatore_linea_persa   = 0
        self._ultima_x_valida         = None
        self._ultima_direzione_errore = None
        self._diramazione_vista_prima = False
        self._buffer_errore_x         = []

    @property
    def contatore_linea_persa(self):
        return self._contatore_linea_persa

    @property
    def ultima_direzione_errore(self):
        return self._ultima_direzione_errore

    @property
    def diramazione_vista_prima(self):
        return self._diramazione_vista_prima

    @diramazione_vista_prima.setter
    def diramazione_vista_prima(self, val):
        self._diramazione_vista_prima = val

    @staticmethod
    def _contorno_e_linea(contorno, hsv, altezza_frame):
        area = cv2.contourArea(contorno)
        if area < 1500:
            return False

        x, y, w, h = cv2.boundingRect(contorno)

        if w == 0 or h == 0:
            return False
        ratio = max(w / h, h / w)
        if ratio < LINEA_ASPECT_RATIO_MIN:
            return False

        hull = cv2.convexHull(contorno)
        area_hull = cv2.contourArea(hull)
        if area_hull == 0:
            return False
        solidita = area / area_hull
        if solidita < LINEA_SOLIDITA_MIN:
            return False

        M = cv2.moments(contorno)
        if M["m00"] == 0:
            return False
        cy = int(M["m01"] / M["m00"])
        if cy < altezza_frame * LINEA_Y_MIN_FRAZIONE:
            return False

        maschera_locale = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(maschera_locale, [contorno], -1, 255, -1)
        saturazione_media = cv2.mean(hsv[:, :, 1], mask=maschera_locale)[0]
        if saturazione_media < LINEA_SATURAZIONE_MIN:
            return False

        return True

    def elabora(self, frame, blu_lower, blu_upper):
        altezza, larghezza, _ = frame.shape
        centro_camera = larghezza // 2
        incrocio_rilevato = False
        errore_x = 0

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, blu_lower, blu_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contorni, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contorni_validi = [
            c for c in contorni
            if self._contorno_e_linea(c, hsv, altezza)
        ]

        if len(contorni_validi) >= 2:
            aree = sorted([cv2.contourArea(c) for c in contorni_validi], reverse=True)
            if aree[1] >= aree[0] * 0.25:
                incrocio_rilevato = True
                self._diramazione_vista_prima = True

        if len(contorni_validi) > 0:
            if self._ultima_x_valida is not None:
                contorno_maggiore = min(contorni_validi,  key=lambda c: int(cv2.moments(c)["m10"] / cv2.moments(c)["m00"]) if cv2.moments(c)["m00"] > 0 else float('inf'))
            else:
                contorno_maggiore = max(contorni_validi, key=cv2.contourArea)

            moments = cv2.moments(contorno_maggiore)
            if moments["m00"] > 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])

                self._contatore_linea_persa = 0
                self._ultima_x_valida = cx
                errore_x_raw = cx - centro_camera

                self._buffer_errore_x.append(errore_x_raw)
                if len(self._buffer_errore_x) > SMOOTHING_BUFFER_SIZE:
                    self._buffer_errore_x.pop(0)
                errore_x = int(np.mean(self._buffer_errore_x))

                if errore_x > TOLLERANZA_OK:
                    self._ultima_direzione_errore = "destra"
                elif errore_x < -TOLLERANZA_OK:
                    self._ultima_direzione_errore = "sinistra"

                cv2.line(frame, (centro_camera, 0), (centro_camera, altezza), (255, 255, 0), 1)
                cv2.line(frame, (0, cy), (larghezza, cy), (255, 255, 0), 1)
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                cv2.line(frame, (cx, cy), (centro_camera, cy), (0, 140, 255), 2)
                cv2.rectangle(frame, (centro_camera - TOLLERANZA_OK, 0),
                              (centro_camera + TOLLERANZA_OK, altezza), (0, 180, 0), 1)
                cv2.rectangle(frame, (centro_camera - TOLLERANZA_MEDIA, 0),
                              (centro_camera + TOLLERANZA_MEDIA, altezza), (0, 220, 220), 1)
                cv2.rectangle(frame, (centro_camera - TOLLERANZA_FORTE, 0),
                              (centro_camera + TOLLERANZA_FORTE, altezza), (0, 0, 200), 1)
                cv2.putText(frame, f"Errore: {errore_x:+d}px", (10, altezza - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)
                cv2.drawContours(frame, contorni_validi, -1, (0, 255, 0), 2)
        else:
            errore_x = None
            self._buffer_errore_x.clear()
            self._contatore_linea_persa += 1

            if (self._contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA
                    and self._ultima_x_valida is not None
                    and self._diramazione_vista_prima):
                incrocio_rilevato = True
            else:
                incrocio_rilevato = False
                if self._contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA:
                    self._diramazione_vista_prima = False

        cv2.putText(frame, f"Linee rilevate: {len(contorni_validi)}", (10, altezza - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame, mask, incrocio_rilevato, errore_x


# =============================================================================
# MacchinaStati — Gestisce la logica decisionale del sistema.
# =============================================================================

class MacchinaStati:

    def __init__(self, voce: MotoreVocale, analizzatore: AnalizzatoreLinea, dati_condivisi):
        self._voce                 = voce
        self._analizzatore         = analizzatore
        self._dati                 = dati_condivisi
        self.stato                 = STATO_NAVIGAZIONE
        self._direzione_da_prendere = None
        self._tempo_inizio_svolta  = 0
        self._mappa_cartelli       = {}
        self._fase_recupero        = 0
        self._correzione_in_corso  = False

    @property
    def mappa_cartelli(self):
        return self._mappa_cartelli

    @property
    def direzione_da_prendere(self):
        return self._direzione_da_prendere

    def aggiorna(self, errore_x, incrocio_rilevato, cartelli_disponibili, larghezza):
        if self.stato == STATO_NAVIGAZIONE:
            self._stato_navigazione(errore_x, incrocio_rilevato, cartelli_disponibili)
        elif self.stato == STATO_LINEA_PERSA:
            self._stato_linea_persa(errore_x)
        elif self.stato == STATO_INCROCIO_ATTESA:
            self._stato_incrocio(errore_x, incrocio_rilevato, cartelli_disponibili)
        elif self.stato == STATO_SVOLTA_AUTOMATICA:
            self._stato_svolta()

    def _stato_navigazione(self, errore_x, incrocio_rilevato, cartelli_disponibili):
        if incrocio_rilevato:
            self._correzione_in_corso = False
            self.stato = STATO_INCROCIO_ATTESA
            self._mappa_cartelli = {}
            self._dati["ocr_cartelli"] = {}
            self._voce.parla("Incrocio rilevato. Rallenta e fermati. Sto leggendo i cartelli.", prioritario=True)
        elif errore_x is not None:
            self._fase_recupero = 0
            abs_err = abs(errore_x)
            lato = "destra" if errore_x > 0 else "sinistra"
            if abs_err <= TOLLERANZA_OK:
                if self._correzione_in_corso:
                    self._correzione_in_corso = False
                    self._voce.parla("Centro ritrovato. Procedi dritto.", prioritario=True)
                else:
                    self._voce.parla("Sei centrato sulla linea. Procedi dritto.")
            else:
                if not self._correzione_in_corso:
                    self._correzione_in_corso = True
                    verso = "sinistra" if lato == "destra" else "destra"
                    self._voce.parla(
                        f"Attenzione! Stai uscendo dalla linea verso {lato}. Correzione immediata verso {verso}.",
                        prioritario=True
                    )
        else:
            if self._analizzatore.contatore_linea_persa >= SOGLIA_FRAME_LINEA_PERSA:
                self._correzione_in_corso = False
                self.stato = STATO_LINEA_PERSA
                self._fase_recupero = 0
                print("[STATO] Transizione a LINEA_PERSA.")

    def _stato_linea_persa(self, errore_x):
        if errore_x is not None:
            self._fase_recupero = 0
            self.stato = STATO_NAVIGAZIONE
            self._voce.parla("Linea ritrovata. Riprendo la navigazione.", prioritario=True)
            return

        dr = self._analizzatore.ultima_direzione_errore
        if dr == "destra":
            dir_recupero, dir_opposta = "destra", "sinistra"
        elif dr == "sinistra":
            dir_recupero, dir_opposta = "sinistra", "destra"
        else:
            dir_recupero, dir_opposta = None, None

        cnt = self._analizzatore.contatore_linea_persa

        if cnt >= SOGLIA_FRAME_LINEA_PERSA and self._fase_recupero < 1:
            self._fase_recupero = 1
            if dir_recupero:
                self._voce.parla(
                    f"Attenzione, linea persa. L'ultima posizione era a {dir_recupero} del tuo campo visivo. "
                    f"Fermati e ruota lentamente verso {dir_opposta}.", prioritario=True)
            else:
                self._voce.parla("Attenzione, linea persa. Fermati e ruota lentamente cercando la linea.", prioritario=True)
            print(f"[RECUPERO FASE 1] contatore={cnt}")

        elif cnt >= SOGLIA_FRAME_PERSA_MEDIA and self._fase_recupero < 2:
            self._fase_recupero = 2
            if dir_recupero:
                self._voce.parla(
                    f"Linea ancora assente. Ruota di trenta gradi verso {dir_recupero} e avanza di un passo.",
                    prioritario=True)
            else:
                self._voce.parla("Linea ancora assente. Ruota di trenta gradi e avanza di un passo.", prioritario=True)
            print(f"[RECUPERO FASE 2] contatore={cnt}")

        elif cnt >= SOGLIA_FRAME_PERSA_TOTALE and self._fase_recupero < 3:
            self._fase_recupero = 3
            if dir_recupero:
                self._voce.parla(
                    f"Percorso completamente smarrito. Torna indietro di due passi, "
                    f"poi ruota verso {dir_recupero} e cerca la linea passando lentamente "
                    f"da {dir_opposta} verso {dir_recupero}.", prioritario=True)
            else:
                self._voce.parla(
                    "Percorso completamente smarrito. Torna indietro di due passi "
                    "e ruota lentamente su te stesso per cercare la linea.", prioritario=True)
            print(f"[RECUPERO FASE 3] contatore={cnt}")

    def _stato_incrocio(self, errore_x, incrocio_rilevato, cartelli_disponibili):
        if not incrocio_rilevato and errore_x is not None:
            self.stato = STATO_NAVIGAZIONE
            self._voce.parla("Linea ripresa. Continuo la navigazione.", prioritario=True)
            return

        if cartelli_disponibili and not self._mappa_cartelli:
            self._mappa_cartelli = cartelli_disponibili.copy()
            elenco = ", oppure ".join(list(cartelli_disponibili.keys()))
            self._voce.parla(f" Ho Letto : {elenco}.", prioritario=True)

            for testo, direzione in cartelli_disponibili.items():
                if direzione in ["DESTRA", "SINISTRA", "DRITTO"]:
                    self._direzione_da_prendere = direzione
                    self._tempo_inizio_svolta   = time.time()
                    self.stato = STATO_SVOLTA_AUTOMATICA
                    print(f"[STATO] Transizione a SVOLTA AUTOMATICA. Direzione: {direzione}")
                    if direzione == "SINISTRA":
                        self._voce.parla(f"Ho letto {testo}. Ruota adesso a sinistra sul posto di novanta gradi.", prioritario=True)
                    elif direzione == "DESTRA":
                        self._voce.parla(f"Ho letto {testo}. Ruota adesso a destra sul posto di novanta gradi.", prioritario=True)
                    else:
                        self._voce.parla(f"Ho letto {testo}. Procedi dritto in avanti per superare l'incrocio.", prioritario=True)
                    break

    def _stato_svolta(self):
        durata = 2.5 if self._direzione_da_prendere in ["DESTRA", "SINISTRA"] else 1.5
        if time.time() - self._tempo_inizio_svolta > durata:
            self._voce.parla("Manovra completata. Riprendo l'assistenza sul percorso lineare.", prioritario=True)
            self._dati["ocr_cartelli"] = {}
            self._mappa_cartelli = {}
            self._analizzatore.diramazione_vista_prima = False
            self.stato = STATO_NAVIGAZIONE
            print("[STATO] Manovra conclusa. Ritorno a NAVIGAZIONE.")


# =============================================================================
# GestoreMonitoraggio — Gestisce la modalità di descrizione ambientale YOLO
# =============================================================================

class GestoreMonitoraggio:

    def __init__(self, voce: MotoreVocale, rilevatore_yolo: RilevatorYOLO, dati_condivisi):
        self._voce            = voce
        self._yolo            = rilevatore_yolo
        self._dati            = dati_condivisi
        self.attivo           = False
        self._ultimo_annuncio = 0
        self._frame_congelato = None
        self._analisi_in_corso = False

    def aggiorna(self, frame, macchina: MacchinaStati):
        if not self.attivo:
            return

        # Se la voce sta ancora parlando, non fare nulla
        if not self._voce.semaforo.is_set():
            return

        # La voce ha finito e c'era un'analisi in corso → sblocca il frame
        if self._analisi_in_corso:
            self._analisi_in_corso = False
            self._frame_congelato  = None   # il loop catturerà un frame fresco
            return

        # Nuova analisi ogni 4 secondi
        now = time.time()
        if now - self._ultimo_annuncio > 4.0:
            descrizione = self._yolo.descrivi_ambiente(frame)
            if descrizione:
                self._voce.parla(descrizione, interrompi_subito=True)
                self._analisi_in_corso = True
            self._ultimo_annuncio = time.time()

        macchina.stato = STATO_NAVIGAZIONE

    def gestisci_tasto(self, tasto, da_voce=False):
        if tasto == ord('w') or tasto == ord('W'):
            if not self.attivo:
                self.attivo = True
                self._ultimo_annuncio = 0
                self._voce.parla(
                    "Monitoraggio ambientale attivato. Metti il telefono parallelo al petto.",
                    interrompi_subito=da_voce,
                    prioritario=not da_voce
                )
                self._voce.semaforo.wait()   
                print("[SISTEMA] Monitoraggio Ambientale Continuo: ACCESO")
        elif tasto == ord('s') or tasto == ord('S'):
            if self.attivo:
                self.attivo = False
                self._frame_congelato = None
                self._voce.parla(
                    "Monitoraggio ambientale disattivato. Metti il telefono parallelo al pavimento. Torno alla navigazione.",
                    interrompi_subito=da_voce,
                    prioritario=not da_voce
                )
                self._voce.semaforo.wait()   
                print("[SISTEMA] Monitoraggio Ambientale Continuo: SPENTO")


# =============================================================================
# OverlayGrafico — Aggiunge le informazioni di stato al frame visualizzato
# =============================================================================

class OverlayGrafico:

    def applica(self, frame, macchina: MacchinaStati, monitoraggio_attivo: bool):
        y = 30  
        riga_h = 35  

        if monitoraggio_attivo:
            cv2.putText(frame, "STATO ASSISTENTE: ASCOLTO VOCALE",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(frame, f"STATO ASSISTENTE: {macchina.stato}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if macchina.stato == STATO_SVOLTA_AUTOMATICA and macchina.direzione_da_prendere:
            y += riga_h
            cv2.putText(frame, f"GUIDA ALLA MANOVRA: {macchina.direzione_da_prendere}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame


# =============================================================================
# main — Inizializza tutti i componenti, avvia i thread e gestisce il loop
# principale di acquisizione video, elaborazione e visualizzazione.
# =============================================================================

def main():
    print("1/3 Caricamento Rete Neurale YOLO...")
    cartella_yolo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Yolo4-tiny")
    yolo_model = YOLO(
        os.path.join(cartella_yolo, "yolov4-tiny.weights"),
        os.path.join(cartella_yolo, "yolov4-tiny.cfg"),
        os.path.join(cartella_yolo, "coco.names")
    )

    print("2/3 Inizializzazione motore OCR (può richiedere qualche secondo)...")
    lettore_ocr = easyocr.Reader(['it', 'en'], gpu=False)

    print("3/3 Connessione allo Stream Video della PWA...")
    FRAME_URL = "http://127.0.0.1:5000/api/leggi-frame"

    frame_attesa = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame_attesa, "In attesa della videocamera (PWA)...", (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame_attesa, "Apri l'app sul telefono e premi 'Inizia'", (40, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    maschera_attesa = np.zeros((480, 640), dtype=np.uint8)

    dati_condivisi = {
        "frame_da_analizzare": None,
        "running":             True,
        "yolo_occupato":       False,
        "yolo_bbox":           [],
        "yolo_labels":         [],
        "yolo_conf":           [],
        "ocr_cartelli":        {},
        "ocr_occupato":        False,
        "frame_linea_persa":   0,
    }

    voce         = MotoreVocale(dati_condivisi)
    yolo_r       = RilevatorYOLO(dati_condivisi, yolo_model)
    ocr_r        = RilevatorOCR(dati_condivisi, lettore_ocr)
    analizzatore = AnalizzatoreLinea()
    macchina     = MacchinaStati(voce, analizzatore, dati_condivisi)
    monitoraggio = GestoreMonitoraggio(voce, yolo_r, dati_condivisi)
    overlay      = OverlayGrafico()

    voce.avvia()
    yolo_r.avvia()
    ocr_r.avvia()

    voce.parla("Dispositivo di assistenza attivo e in ascolto. Puoi iniziare a camminare.", prioritario=True)

    while dati_condivisi["running"]:

        # ── 1. LETTURA TASTO E COMANDI VOCALI ──────────────────────────────────
        tasto   = cv2.waitKey(1) & 0xFF
        da_voce = False
        cmd     = voce.comando_remoto
        if cmd:
            if   cmd in ['W', 'W_VOICE']: tasto = ord('w'); da_voce = True
            elif cmd in ['S', 'S_VOICE']: tasto = ord('s'); da_voce = True
            elif cmd in ['Q', 'Q_VOICE']: tasto = ord('q'); da_voce = True
            if da_voce:
                voce.svuota_coda()

        monitoraggio.gestisci_tasto(tasto, da_voce=da_voce)

        if tasto == ord('q'):
            break

        # ── 2. ACQUISIZIONE FRAME DAL TELEFONO ─────────────────────────────────
        frame_corrente = None
        try:
            r = requests.get(FRAME_URL, timeout=0.5)
            if r.status_code == 200:
                nparr = np.frombuffer(r.content, np.uint8)
                frame_corrente = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except requests.exceptions.RequestException:
            pass

        # Schermata di attesa se il telefono non ha ancora inviato video
        if frame_corrente is None:
            cv2.imshow("Maschera", maschera_attesa)
            cv2.imshow("Navigation System", frame_attesa)
            continue

        # ── 3. MODALITÀ MONITORAGGIO AMBIENTALE (frame congelato) ──────────────
        if monitoraggio.attivo:

            # aggiorna() ha resettato _frame_congelato a None → vogliamo un frame fresco
            # lo congeliamo con il frame appena arrivato in questo ciclo
            if monitoraggio._frame_congelato is None:
                monitoraggio._frame_congelato = frame_corrente.copy()
                # Resetta anche l'ultimo annuncio così YOLO parte subito sul frame nuovo
                monitoraggio._ultimo_annuncio = 0

            # Chiama aggiorna solo se il frame è congelato (non None)
            monitoraggio.aggiorna(monitoraggio._frame_congelato, macchina)

            # Mostra il frame congelato
            frame_display  = monitoraggio._frame_congelato.copy()
            frame_finale   = overlay.applica(frame_display, macchina, True)
            maschera_vuota = np.zeros(
                (frame_display.shape[0], frame_display.shape[1]), dtype=np.uint8
            )
            cv2.imshow("Maschera", maschera_vuota)
            cv2.imshow("Navigation System", frame_finale)
            continue

        # ── 4. MODALITÀ NAVIGAZIONE NORMALE ────────────────────────────────────

        # Resetta il frame congelato quando il monitoraggio è spento
        monitoraggio._frame_congelato = None

        # Aggiorna il frame da analizzare per i thread YOLO e OCR
        if not dati_condivisi["yolo_occupato"] and not dati_condivisi["ocr_occupato"]:
            dati_condivisi["frame_da_analizzare"] = frame_corrente.copy()

        # Disegna i bounding box YOLO sul frame corrente
        frame_corrente, _ = yolo_r.disegna_su_frame(frame_corrente)

        # Analisi linee blu e stato macchina
        cartelli_disponibili = dati_condivisi["ocr_cartelli"].copy()
        frame_linee, maschera_linee, incrocio_rilevato, errore_x = analizzatore.elabora(
            frame_corrente.copy(), BLU_LOWER, BLU_UPPER
        )
        macchina.aggiorna(errore_x, incrocio_rilevato, cartelli_disponibili,
                          larghezza=frame_corrente.shape[1])

        # Visualizzazione
        frame_finale = overlay.applica(frame_linee, macchina, False)
        cv2.imshow("Maschera", maschera_linee)
        cv2.imshow("Navigation System", frame_finale)

    dati_condivisi["running"] = False
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()