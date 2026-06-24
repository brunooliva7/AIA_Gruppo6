import subprocess
import time
import sys
import os

def main():
    print("========================================")
    print(" Avvio del Sistema di Navigazione Voice ")
    print("========================================\n")
    
    percorso_cartella = os.path.dirname(os.path.abspath(__file__))
    
    script_web = os.path.join(percorso_cartella, "SitoWebDialogoVoice.py")
    script_nav = os.path.join(percorso_cartella, "ProvaB (1).py")
    
    print("[1/2] Avvio del server web Flask...")
    # Avvia il server Flask
    process_web = subprocess.Popen([sys.executable, script_web])
    
    # Attendiamo 3 secondi per permettere al server di inizializzarsi
    time.sleep(3)
    
    print("\n[2/2] Avvio del modulo di navigazione OpenCV...")
    # Avvia il processo ProvaB
    process_nav = subprocess.Popen([sys.executable, script_nav])
    
    print("\n>>> Tutti i moduli sono stati avviati con successo! <<<")
    print("Premi CTRL+C per terminare entrambi i processi.\n")
    
    try:
        # Mantiene in vita il processo main finché il modulo di navigazione è attivo
        process_nav.wait()
        
        # Appena ProvaB si chiude, terminiamo anche il server Flask
        print("\n\n[INFO] Chiusura in corso...")
        process_web.terminate()
        process_web.wait()
        print("[INFO] Tutti i processi sono stati terminati.")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Chiusura forzata in corso...")
        process_web.terminate()
        process_nav.terminate()
        process_web.wait()
        process_nav.wait()
        print("[INFO] Tutti i processi sono stati terminati.")

if __name__ == "__main__":
    main()
