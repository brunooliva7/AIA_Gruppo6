import cv2
import numpy as np

def riconoscimento_linee_guida():

    video = cv2.VideoCapture(1)

    BLU_LOWER = np.array([100, 150, 0])
    BLU_UPPER = np.array([140, 255, 255])

    while video.isOpened():
        #ret : booleano che indica se la lettura del frame è avvenuta con successo
        #frame : il frame catturato dalla videocamera
        ret, frame = video.read()
        if not ret:
            print("Errore nella lettura del video")
            break

        #OTTENERE DIMENSIONI DEL FRAME (RISOLUZIONE DI RUN TIME)
        altezza, larghezza, _ = frame.shape
        centro_camera = larghezza // 2
        punto_inizio_linea_guida = (centro_camera, altezza//2)


        #PRE-PROCESSING DEL FRAME PER ELIMINARE RUMORE (imperfezioni pavimento)
        blur  = cv2.GaussianBlur(frame, (5, 5), 0)

        #CONVERSIONE DEL FRAME IN SPAZIO COLORI HSV PER ISOLARE IL COLORE BLU DELLE LINEE GUIDA
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #CREAZIONE DI UNA MASCHERA PER ISOLARE LE LINEE GUIDA BLU
        mask = cv2.inRange(hsv, BLU_LOWER, BLU_UPPER)

        #CALCOLO DEI MOMENTI DELL'IMMAGINE PER TROVARE IL CENTRO DELLA MASCHERA
        moments = cv2.moments(mask)

        if moments["m00"] > 0:      #m00 :  momento di ordine zero, rappresenta l'area totale dell'oggetto rilevato, espresso in pixel
            centro_linea_guida_x = int(moments["m10"] / moments["m00"])  #m10 : momento di ordine uno, rappresenta la somma dei prodotti delle coordinate x dei pixel per i loro valori di intensità
            centro_linea_guida_y = int(moments["m01"] / moments["m00"])  #m01 : momento di ordine uno, rappresenta la somma dei prodotti delle coordinate y dei pixel per i loro valori di intensità

            #DISEGNO DI UN CERCHIO E DELLA LINEA CHE CONNETTE IL CENTRO DELLA CAMERA AL CENTRO DELLE LINEE GUIDA
            cv2.circle(frame, (centro_linea_guida_x, centro_linea_guida_y), 10, (0, 0, 255), -1)
            cv2.line(frame, punto_inizio_linea_guida, (centro_linea_guida_x, centro_linea_guida_y), (0, 255, 0), 2)
            
            #TOLLERANZA PER LA DIREZIONE (SE LA LINEA GUIDA È VICINA AL CENTRO DELLA CAMERA, CONSIDERALA ALLINEATA)
            tolleranza = 50
            errore_x = centro_linea_guida_x - centro_camera

            if errore_x > tolleranza:
                comando_motori = "Curva a Destra"
            elif errore_x < -tolleranza:
                comando_motori = "Curva a Sinistra"
            else:
                comando_motori = "Avanti Dritto"
            
            #INVIO COMANDI A PEPPER VIA WEBHOOK
            print(f"Comando per i motori: {comando_motori}")
        
        else:
            cv2.putText(frame, "LINEA PERDUTA - STOP", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            #se linea guida è persa, invia comando di stop a Pepper
        
        #VISUALIZZAZIONE DEL FRAME ELABORATO
        cv2.imshow("Riconoscimento Linee Guida", frame)
        cv2.imshow("Maschera Linee Guida", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    riconoscimento_linee_guida()