import os
import cv2
import numpy as np
import threading
import time
import easyocr
import queue
import speech_recognition as sr
from cvlib.object_detection import YOLO
import winsound


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

SOGLIA_MINIMA_PIXEL       = 3000
SOGLIA_Y_INCROCIO         = 320
SOGLIA_FRAME_LINEA_PERSA  = 15
SOGLIA_FRAME_PERSA_MEDIA  = 45
SOGLIA_FRAME_PERSA_TOTALE = 90

TOLLERANZA_OK    = 60
TOLLERANZA_MEDIA = 130
TOLLERANZA_FORTE = 210
SMOOTHING_BUFFER_SIZE = 5

INTERVALLO_VOCE = 1.8

STATO_NAVIGAZIONE       = "NAVIGAZIONE"
STATO_INCROCIO_ATTESA   = "INCROCIO_ATTESA"
STATO_SVOLTA_AUTOMATICA = "SVOLTA_AUTOMATICA"
STATO_LINEA_PERSA       = "LINEA_PERSA"


# =============================================================================
# MotoreVocale — Gestisce la sintesi vocale TTS tramite SAPI in un thread
# dedicato. Supporta messaggi normali, prioritari (svuotano la coda) e
# interruzione immediata via SpVoice.Skip().
# =============================================================================

class MotoreVocale:

    def __init__(self, dati_condivisi):
        self._dati           = dati_condivisi
        self._coda           = queue.Queue()
        self._semaforo       = threading.Event()
        self._semaforo.set()
        self._sapi           = None
        self._lock           = threading.Lock()
        self._flag_interrompi = threading.Event()
        self._ultimo_msg     = ""
        self._ultimo_tempo   = 0.0

    def avvia(self):
        threading.Thread(target=self._loop_tts, daemon=True).start()

    def _loop_tts(self):
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        with self._lock:
            self._sapi = win32com.client.Dispatch("SAPI.SpVoice")
            self._sapi.Rate = 1
        print("[THREAD VOCE UNIFICATO] Pronto.")
        while self._dati["running"]:
            try:
                testo = self._coda.get(timeout=0.5)
                if self._flag_interrompi.is_set():
                    self._flag_interrompi.clear()
                    self._coda.task_done()
                    continue
                self._semaforo.clear()
                print(f"\n>>> [SISTEMA PARLA]: {testo} <<<\n")
                self._flag_interrompi.clear()
                with self._lock:
                    self._sapi.Speak(testo, 0)
                self._semaforo.set()
                self._coda.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERRORE AUDIO] {e}")
                self._semaforo.set()

    def _stop_ora(self):
        self._flag_interrompi.set()
        with self._lock:
            if self._sapi is not None:
                try:
                    self._sapi.Skip("Sentence", 999)
                except Exception:
                    pass
        while not self._coda.empty():
            try:
                self._coda.get_nowait()
                self._coda.task_done()
            except queue.Empty:
                break
        self._semaforo.set()

    def parla(self, testo, prioritario=False, interrompi=False):
        now = time.time()
        if interrompi:
            self._stop_ora()
            self._flag_interrompi.clear()
            self._coda.put(testo)
            self._ultimo_msg   = ""
            self._ultimo_tempo = 0.0
        elif prioritario:
            while not self._coda.empty():
                try:
                    self._coda.get_nowait()
                    self._coda.task_done()
                except queue.Empty:
                    break
            self._semaforo.clear()
            self._coda.put(testo)
            self._ultimo_msg   = ""
            self._ultimo_tempo = 0.0
        else:
            if testo != self._ultimo_msg or (now - self._ultimo_tempo) > INTERVALLO_VOCE:
                self._semaforo.clear()
                self._coda.put(testo)
                self._ultimo_msg   = testo
                self._ultimo_tempo = now

    def svuota_coda(self):
        while not self._coda.empty():
            try:
                self._coda.get_nowait()
                self._coda.task_done()
            except queue.Empty:
                break

    @property
    def semaforo(self):
        return self._semaforo


# =============================================================================
# RilevatorYOLO — Esegue il rilevamento ostacoli in un thread dedicato.
# Aggiorna continuamente bbox, label e confidenze nel dizionario condiviso.
# =============================================================================

class RilevatorYOLO:

    def __init__(self, dati_condivisi, yolo_model):
        self._dati  = dati_condivisi
        self._model = yolo_model

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
    def calcola_distanza(altezza_reale_cm, altezza_pixel):
        if altezza_pixel == 0:
            return 0
        return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

    def disegna_su_frame(self, frame):
        ostacolo_vicino = False
        for i, label in enumerate(self._dati["yolo_labels"]):
            try:
                x_min, y_min, x_max, y_max = self._dati["yolo_bbox"][i]
                alt_pixel   = y_max - y_min
                distanza_cm = self.calcola_distanza(ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT), alt_pixel)
                cat = "PERSONA" if label == 'person' else "OGGETTO"
                if distanza_cm < DISTANZA_ALLARME:
                    ostacolo_vicino = True
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                    cv2.putText(frame, f"ALLARME {cat} ({distanza_cm/100:.1f}m)",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.putText(frame, f"{cat} ({distanza_cm/100:.1f}m)",
                                (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except IndexError:
                pass
        return frame, ostacolo_vicino

    def descrivi_ambiente(self, frame):
        labels_viste = list(self._dati.get("yolo_labels", []))
        bbox_visti   = list(self._dati.get("yolo_bbox", []))
        if not labels_viste:
            print("[AMBIENTE] Nessun oggetto rilevato nei thread di background...")
            return None
        oggetti_descritti = []
        for i, label in enumerate(labels_viste):
            try:
                x_min, y_min, x_max, y_max = bbox_visti[i]
                alt_pixel   = y_max - y_min
                distanza_cm = self.calcola_distanza(ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT), alt_pixel)
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
                oggetti_descritti.append(f"{nome_ita} {lato}, a circa {distanza_m:.1f} metri")
            except (IndexError, Exception):
                pass
        if oggetti_descritti:
            return "Nell'ambiente rilevo: " + "; ".join(oggetti_descritti) + "."
        return "Non riesco a stimare la distanza degli oggetti rilevati."


# =============================================================================
# RilevatorOCR — Legge testi da cartelli in un thread dedicato e aggiorna
# il dizionario condiviso con le direzioni riconosciute (DESTRA/SINISTRA/DRITTO).
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
# AnalizzatoreLinea — Elabora ogni frame per rilevare la linea blu centrale,
# calcola l'errore di posizione (con smoothing) e individua gli incroci.
# Disegna le zone di confidenza sul frame.
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

    def elabora(self, frame, blu_lower, blu_upper):
        altezza, larghezza, _ = frame.shape
        centro_camera = larghezza // 2
        incrocio_rilevato = False
        errore_x = 0

        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, blu_lower, blu_upper)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contorni, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contorni_validi = [c for c in contorni if cv2.contourArea(c) > 1500]

        if len(contorni_validi) >= 2:
            aree = sorted([cv2.contourArea(c) for c in contorni_validi], reverse=True)
            if aree[1] >= aree[0] * 0.25:
                incrocio_rilevato = True
                self._diramazione_vista_prima = True

        if len(contorni_validi) > 0:
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
# MacchinaStati — Gestisce la logica decisionale del sistema (navigazione,
# linea persa, attesa incrocio, svolta automatica) e genera i comandi vocali.
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
            lato = "destra" if errore_x < 0 else "sinistra"
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
                    f"Fermati e ruota lentamente verso {dir_recupero}.", prioritario=True)
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
        durata = 4.5 if self._direzione_da_prendere in ["DESTRA", "SINISTRA"] else 2.5
        if time.time() - self._tempo_inizio_svolta > durata:
            self._voce.parla("Manovra completata. Riprendo l'assistenza sul percorso lineare.", prioritario=True)
            self._dati["ocr_cartelli"] = {}
            self._mappa_cartelli = {}
            self._analizzatore.diramazione_vista_prima = False
            self.stato = STATO_NAVIGAZIONE
            print("[STATO] Manovra conclusa. Ritorno a NAVIGAZIONE.")


# =============================================================================
# GestoreMonitoraggio — Gestisce la modalità di descrizione ambientale YOLO
# attivabile con W/S da tastiera. In questa modalità la macchina a stati è
# sospesa e ogni 7 secondi viene letto l'ambiente circostante.
# =============================================================================

class GestoreMonitoraggio:

    def __init__(self, voce: MotoreVocale, rilevatore_yolo: RilevatorYOLO, dati_condivisi):
        self._voce            = voce
        self._yolo            = rilevatore_yolo
        self._dati            = dati_condivisi
        self.attivo           = False
        self._ultimo_annuncio = 0

    def aggiorna(self, frame, macchina: MacchinaStati):
        if not self.attivo:
            return
        now = time.time()
        if now - self._ultimo_annuncio > 7.0:
            descrizione = self._yolo.descrivi_ambiente(frame)
            if descrizione:
                self._voce.parla(descrizione, prioritario=True)
            self._voce.semaforo.wait()
            self._ultimo_annuncio = now

        macchina.stato = STATO_NAVIGAZIONE
        self._voce.svuota_coda()

    def gestisci_tasto(self, tasto):
        if tasto == ord('w') or tasto == ord('W'):
            if not self.attivo:
                self.attivo = True
                self._ultimo_annuncio = 0
                self._voce.parla("Monitoraggio ambientale attivato. Metti il telefono parallelo al petto.", prioritario=True)
                print("[SISTEMA] Monitoraggio Ambientale Continuo: ACCESO")
        elif tasto == ord('s') or tasto == ord('S'):
            if self.attivo:
                self.attivo = False
                self._voce.parla("Monitoraggio ambientale disattivato. Metti il telefono parallelo al pavimento. Torno alla navigazione.", prioritario=True)
                print("[SISTEMA] Monitoraggio Ambientale Continuo: SPENTO")


# =============================================================================
# OverlayGrafico — Aggiunge le informazioni di stato al frame visualizzato
# a schermo (stato corrente, direzione manovra in corso).
# =============================================================================

class OverlayGrafico:

    def applica(self, frame, macchina: MacchinaStati, monitoraggio_attivo: bool):
        if monitoraggio_attivo:
            cv2.putText(frame, "STATO ASSISTENTE: ASCOLTO VOCALE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            cv2.putText(frame, f"STATO ASSISTENTE: {macchina.stato}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        if macchina.stato == STATO_SVOLTA_AUTOMATICA and macchina.direzione_da_prendere:
            cv2.putText(frame, f"GUIDA ALLA MANOVRA: {macchina.direzione_da_prendere}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
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

    print("3/3 Accensione Telecamera...")
    video = cv2.VideoCapture(1)

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

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("[ERRORE] Impossibile acquisire il video dalla telecamera.")
            break

        if not dati_condivisi["yolo_occupato"] and not dati_condivisi["ocr_occupato"]:
            dati_condivisi["frame_da_analizzare"] = frame.copy()

        frame, _ = yolo_r.disegna_su_frame(frame)

        if monitoraggio.attivo:
            monitoraggio.aggiorna(frame, macchina)

        dati_condivisi["frame_da_analizzare"] = frame.copy()
        cartelli_disponibili = dati_condivisi["ocr_cartelli"].copy()

        frame_linee, maschera_linee, incrocio_rilevato, errore_x = analizzatore.elabora(
            frame.copy(), BLU_LOWER, BLU_UPPER
        )

        if not monitoraggio.attivo:
            macchina.aggiorna(errore_x, incrocio_rilevato, cartelli_disponibili, larghezza=frame.shape[1])

        tasto = cv2.waitKey(1) & 0xFF
        monitoraggio.gestisci_tasto(tasto)

        if tasto == ord('q'):
            break

        cv2.putText(frame_linee, f"STATO: {macchina.stato}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        frame_finale = overlay.applica(frame_linee, macchina, monitoraggio.attivo)
        cv2.imshow("Maschera", maschera_linee)
        cv2.imshow("Navigation System", frame_finale)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dati_condivisi["running"] = False
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()