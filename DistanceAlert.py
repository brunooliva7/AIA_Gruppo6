import cv2
import cvlib as cv
from cvlib.object_detection import YOLO
import time
import os

# 1. PARAMETRI DI CALIBRAZIONE
# Questo valore dipende dalla tua webcam. 
# Se la distanza calcolata è sbagliata, aumenta o diminuisci questo numero.
FOCAL_LENGTH = 300

# Distanza limite per far scattare l'allarme (in centimetri)
DISTANZA_ALLARME = 150 

# 2. DIZIONARIO ALTEZZE REALI (in centimetri)
# YOLO riconosce 80 oggetti. Qui inseriamo le altezze medie di alcuni.
# Puoi aggiungere tutti quelli che ti servono dalla lista COCO.
# DIZIONARIO ALTEZZE REALI PER AMBIENTI CHIUSI E UFFICIO (in centimetri)
ALTEZZE_REALI_CM = {
    # Persone e Animali
    'person': 170,       # Persona adulta
    'dog': 60,           # Cane di taglia media
    'cat': 30,           # Gatto
    'bird': 15,          # Uccellino/Pappagallo

    # Arredamento principale
    'chair': 90,         # Sedia (altezza schienale)
    'sofa': 85,          # Divano
    'bed': 60,           # Letto (altezza materasso)
    'dining table': 75,  # Tavolo
    'toilet': 80,        # WC (con cassetta)
    'refrigerator': 180, # Frigorifero
    
    # Elettronica e Scrivania
    'tv': 65,            # Monitor o TV media
    'laptop': 25,        # Portatile aperto
    'mouse': 4,          # Mouse del computer
    'keyboard': 15,      # Tastiera (profondità vista dall'alto)
    'cell phone': 15,    # Smartphone
    'remote': 20,        # Telecomando
    'clock': 30,         # Orologio da parete

    # Piccoli elettrodomestici da cucina o bagno
    'microwave': 30,     # Microonde
    'oven': 85,          # Forno
    'toaster': 20,       # Tostapane
    'sink': 90,          # Lavandino (altezza da terra)
    'hair drier': 25,    # Asciugacapelli

    # Oggetti di uso comune e cucina
    'bottle': 25,        # Bottiglia d'acqua
    'wine glass': 20,    # Calice
    'cup': 10,           # Tazza
    'fork': 20,          # Forchetta
    'knife': 22,         # Coltello
    'spoon': 15,         # Cucchiaio
    'bowl': 10,          # Ciotola
    'book': 25,          # Libro
    'vase': 25,          # Vaso
    'potted plant': 60,  # Pianta da interno media

    # Accessori personali
    'backpack': 45,      # Zaino
    'umbrella': 80,      # Ombrello chiuso
    'handbag': 30,       # Borsa
    'tie': 45,           # Cravatta
    'suitcase': 65,      # Valigia

    # Varie
    'scissors': 15,      # Forbici
    'teddy bear': 40,    # Peluche
    'toothbrush': 20     # Spazzolino
}

# Altezza di default se l'oggetto rilevato non è nel dizionario
ALTEZZA_DEFAULT = 50 

def calcola_distanza(altezza_reale_cm, altezza_pixel):
    """Calcola la distanza usando i triangoli simili"""
    if altezza_pixel == 0:
        return 0
    return (altezza_reale_cm * FOCAL_LENGTH) / altezza_pixel

def invia_alert(etichetta, distanza):
    """Simula l'invio di un segnale di allarme (es. verso MQTT)"""
    # Qui potresti inserire il codice per inviare il segnale a Dialogflow, a un broker MQTT o a un robot
    print(f"⚠️ ALLARME! Oggetto '{etichetta}' rilevato a soli {distanza:.2f} metri! ⚠️")

def main():

    cartella_script = os.path.dirname(os.path.abspath(__file__))
    cartella_yolo = os.path.join(cartella_script, "Yolo4-tiny") # Nome della nuova cartella

    yolo_weights = os.path.join(cartella_yolo, "yolov4-tiny.weights")
    yolo_cfg = os.path.join(cartella_yolo, "yolov4-tiny.cfg")
    coco_names = os.path.join(cartella_yolo, "coco.names")

    print("Caricamento rete neurale dalla cartella 'modelli_yolo'...")
    try:
        # Inizializziamo il modello YOLO manualmente dai file locali
        yolo_model = YOLO(yolo_weights, yolo_cfg, coco_names)
    except Exception as e:
        print(f"Errore: Impossibile trovare i file in {cartella_yolo}. Assicurati di averli spostati!")
        return
    

    # Apri la webcam (0 è la webcam predefinita)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Errore: Impossibile accedere alla webcam.")
        return

    print("Avvio sistema di rilevamento e stima distanza...")
    print("Premi 'q' per uscire.")

    # Variabile per non inondare il terminale di alert (1 alert al secondo)
    ultimo_alert = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Rilevamento oggetti con YOLOv4-tiny (ottimo per il tempo reale)
        bbox, labels, conf = yolo_model.detect_objects(frame)

        # Analizziamo ogni oggetto trovato
        for i, label in enumerate(labels):
            # Estraiamo le coordinate: [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox[i]
            
            # Calcoliamo quanti pixel è alto l'oggetto sullo schermo
            altezza_pixel = y_max - y_min
            
            # Recuperiamo l'altezza reale stimata dal dizionario
            altezza_reale = ALTEZZE_REALI_CM.get(label, ALTEZZA_DEFAULT)
            
            # Calcoliamo la distanza in centimetri e la convertiamo in metri
            distanza_cm = calcola_distanza(altezza_reale, altezza_pixel)
            distanza_m = distanza_cm / 100

            # --- LOGICA DI ALLARME ---
            if distanza_cm < DISTANZA_ALLARME:
                # Disegna un box ROSSO e testo di allarme
                colore_box = (0, 0, 255) # BGR
                testo = f"ALLARME! {label}: {distanza_m:.2f}m"
                
                # Invia segnale di alert se è passato almeno 1 secondo dal precedente
                if time.time() - ultimo_alert > 1:
                    invia_alert(label, distanza_m)
                    ultimo_alert = time.time()
            else:
                # Disegna un box VERDE se è a distanza di sicurezza
                colore_box = (0, 255, 0)
                testo = f"{label}: {distanza_m:.2f}m"

            # Disegno manuale del box e del testo per avere colori condizionali
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), colore_box, 2)
            
            # Sfondo nero per il testo per renderlo leggibile
            (w_txt, h_txt), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x_min, y_min - 20), (x_min + w_txt, y_min), (0,0,0), -1)
            cv2.putText(frame, testo, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colore_box, 2)

        # Mostra il flusso video
        cv2.imshow("Sistema Anti-Intrusione (Stima Distanza)", frame)

        # Premi 'q' per uscire dal ciclo
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascio delle risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()