import speech_recognition as sr
import pyttsx3
import queue
import threading
import time
import winsound

WAKE_WORD = "assistente"

# --- STRUMENTI DI SINCRONIZZAZIONE (CONCORRENZA) ---
coda_comandi_utente = queue.Queue() # Passa i dati dall'Orecchio al Main
coda_testo_da_leggere = queue.Queue() # Passa i testi dal Main/Orecchio alla Bocca

# IL SEMAFORO! (Event)
# False = Rosso (La bocca sta parlando, l'orecchio deve aspettare)
# True = Verde (Silenzio, l'orecchio può registrare)
semaforo_voce = threading.Event()
semaforo_voce.set() # All'avvio è verde


# --- FUNZIONE DI UTILITÀ: FEEDBACK CHIUSURA ---
def esegui_bip_chiusura():
    """Emette un doppio tono discendente per indicare che il microfono è chiuso."""
    winsound.Beep(600, 150)
    winsound.Beep(450, 150)
    print("🔒 [SESSIONE CHIUSA] Microfono disattivato. Ritorno in background.\n")


# --- 1. IL THREAD DELLA BOCCA (Output Vocale Asincrono) ---
def thread_bocca_tts():
    motore_voce = pyttsx3.init()
    motore_voce.setProperty('rate', 170)
    
    while True:
        testo = coda_testo_da_leggere.get() # Aspetta finché non c'è qualcosa da leggere
        
        semaforo_voce.clear() # 🔴 Mette il semaforo ROSSO (Sto iniziando a parlare!)
        
        print(f"🤖 Sistema: {testo}")
        motore_voce.say(testo)
        motore_voce.runAndWait() # Questo blocca solo la Bocca, non blocca più l'Orecchio!
        
        semaforo_voce.set() # 🟢 Mette il semaforo VERDE (Ho finito di parlare!)


# --- 2. IL THREAD DELL'ORECCHIO (Con Adattamento Ambientale Continuo) ---
def thread_ascolto_google():
    riconoscitore = sr.Recognizer()
    microfono = sr.Microphone()
    
    riconoscitore.pause_threshold = 0.5
    riconoscitore.non_speaking_duration = 0.4
    
    with microfono as source:
        print("🎧 [MICROFONO] Calibrazione iniziale al boot...")
        riconoscitore.adjust_for_ambient_noise(source, duration=2.0)
        
        print(f"🎧 Pronti. Di' '{WAKE_WORD}' per attivare.")
        
        while True:
            # --- FASE 1: ASCOLTO PASSIVO (Silenziato per gli errori) ---
            riconoscitore.dynamic_energy_threshold = True
            
            try:
                audio_sveglia = riconoscitore.listen(source, timeout=None, phrase_time_limit=3)
                testo_sveglia = riconoscitore.recognize_google(audio_sveglia, language="it-IT").lower()
            except Exception:
                # 🤫 SILENZIO ASSOLUTO: Ignoriamo tosse, rumori di fondo e piccoli errori di rete.
                # Torna semplicemente all'inizio del ciclo a riascoltare.
                continue 
            
            # --- FASE 2: ATTIVAZIONE E COMANDO ---
            if WAKE_WORD in testo_sveglia:
                print("\n🟢 [TRIGGER] Parola d'ordine intercettata!")
                
                riconoscitore.dynamic_energy_threshold = False
                coda_testo_da_leggere.put("Sono in ascolto.")
                
                time.sleep(0.1) 
                semaforo_voce.wait() 
                
                winsound.Beep(800, 200)
                print("🎤 [REGISTRAZIONE] PARLA ORA!...")
                
                try:
                    # Registra il vero comando dell'utente
                    audio_comando = riconoscitore.listen(source, timeout=5, phrase_time_limit=None)
                    print("☁️ [CLOUD] Elaborazione in corso...")
                    comando_finale = riconoscitore.recognize_google(audio_comando, language="it-IT").lower()
                    
                    print(f"✅ [TRADOTTO] Hai detto: '{comando_finale}'")
                    coda_comandi_utente.put(comando_finale)
                    
                # ⚠️ QUESTI ERRORI SCATTANO SOLO SE IL COMANDO FALLISCE DOPO IL BIP
                except sr.WaitTimeoutError:
                    print("⏳ [TIMEOUT] Nessun comando rilevato.")
                    coda_testo_da_leggere.put("Non ho sentito nulla. Annullamento.")
                    semaforo_voce.wait() 
                    esegui_bip_chiusura() 
                    
                except sr.UnknownValueError:
                    print("🤷‍♂️ [STT ERROR] Audio incomprensibile.")
                    coda_testo_da_leggere.put("Non sono riuscito a decifrare l'audio.")
                    semaforo_voce.wait()
                    esegui_bip_chiusura()
                    
                except sr.RequestError as e:
                    print(f"❌ [ERRORE RETE] Impossibile contattare Google: {e}")
                    coda_testo_da_leggere.put("Errore di rete.")
                    semaforo_voce.wait()
                    esegui_bip_chiusura()


# --- 3. IL THREAD PRINCIPALE (YOLO E MACCHINA A STATI) ---
if __name__ == "__main__":
    
    # Avvia i due lavoratori in background
    threading.Thread(target=thread_bocca_tts, daemon=True).start()
    threading.Thread(target=thread_ascolto_google, daemon=True).start()
    
    coda_testo_da_leggere.put("Sistema avviato e in ascolto.")
    
    while True:
        try:
            # Simulazione frame rate di YOLO...
            time.sleep(0.1) 
            
            if not coda_comandi_utente.empty():
                comando = coda_comandi_utente.get()
                print(f"⚙️ [MAIN] Ricevuto da eseguire: '{comando}'")
                
                # --- LOGICA DI NAVIGAZIONE E GESTIONE INTENTI ---
                if "ambiente" in comando or "intorno" in comando:
                    coda_testo_da_leggere.put("Avvio l'esplorazione ambientale. Muovi il busto per lo scouting.")
                    # Dopo aver risposto, chiudiamo la sessione audio
                    semaforo_voce.wait()
                    esegui_bip_chiusura()
                    
                elif "fermati" in comando:
                    coda_testo_da_leggere.put("Arresto immediato.")
                    semaforo_voce.wait()
                    break
                    
                elif "grazie" in comando or "annulla" in comando or "chiudi" in comando:
                    coda_testo_da_leggere.put("Va bene, disattivo il microfono.")
                    semaforo_voce.wait()
                    esegui_bip_chiusura()
                    
                else:
                    # Simulazione della risposta statica di Dialogflow per mancata comprensione
                    coda_testo_da_leggere.put("Non ho capito, puoi ripetere?")
                    semaforo_voce.wait()
                    esegui_bip_chiusura()
                    
        except KeyboardInterrupt:
            print("\n🛑 Interruzione manuale. Spegnimento in corso...")
            break