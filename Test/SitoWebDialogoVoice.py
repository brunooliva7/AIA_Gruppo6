import os
from flask import Flask, request, jsonify, render_template_string
# NOTA: È necessario installare la libreria di Google Cloud per Dialogflow:
# pip install google-cloud-dialogflow
from google.cloud import dialogflow

app = Flask(__name__)

# ==========================================
# CONFIGURAZIONE CONNESSIONE DIALOGFLOW
# ==========================================
# Sostituisci questi valori con i dati del tuo progetto
PROJECT_ID = "trackbuddy-tpbx"  
SERVICE_ACCOUNT_FILE = "chiave_dialogFlow.json"

# Imposta la variabile d'ambiente per dire a Google dove trovare le credenziali
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# ID sessione fittizio (in una vera app potresti generarlo per ogni utente)
SESSION_ID = "sessione-vocale-test-123"
LANGUAGE_CODE = "it-IT"

# Coda globale per i messaggi vocali provenienti da ProvaB
messaggi_pendenti = []
taglia_audio_flag = False

# Comando pendente in arrivo dall'interfaccia web verso ProvaB
comando_pendente = None


# ==========================================
# INTERFACCIA WEB (HTML + JAVASCRIPT)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Centro di Controllo Assistenza</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 20px auto; text-align: center; background-color: #121212; color: #ffffff; }
        button { padding: 15px 20px; font-size: 18px; margin: 10px; cursor: pointer; border: none; border-radius: 8px; font-weight: bold; transition: opacity 0.2s; width: 90%; max-width: 400px; }
        button:hover { opacity: 0.8; }
        
        #btnRecord { background-color: #ff4757; color: white; }
        #btnRecord:disabled { background-color: #ff475780; cursor: not-allowed; }
        #btnStop { background-color: #2ed573; color: white; display: none; }
        
        #status { font-size: 1.2em; margin: 20px 0; color: #a4b0be; }
        #result { background-color: #1e272e; padding: 20px; border-radius: 10px; text-align: left; margin-top: 20px; display: none; }
        .highlight { color: #2ed573; font-weight: bold; }
        
        hr { border-color: #2f3542; margin: 30px 0; }
    </style>
</head>
<body>
    <!-- Overlay iniziale per abilitare l'audio policy del browser -->
    <div id="initOverlay" style="position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:9999; display:flex; flex-direction:column; justify-content:center; align-items:center;">
        <h2>Benvenuto nell'Assistente</h2>
        <p>Premi il pulsante per abilitare l'audio e iniziare.</p>
        <button id="btnInit" style="background-color: #2ed573; color: white; padding: 20px 40px; font-size: 24px;">Inizia</button>
    </div>

    <h1>Centro di Controllo</h1>
    <p>Interfaccia Telecomando Web</p>
    
    <button id="btnRecord">🔴 Parla con l'Assistente</button>
    <button id="btnStop">⬆️ Ferma e Invia</button>
    
    <div id="status">Pronto per registrare.</div>
    <div id="result"></div>

    <script>
        let isRecordingOrWaiting = false;
        let pollingInterval = null;

        // ======= FUNZIONALITÀ TEXT-TO-SPEECH (Voce) =======
        // BUGFIX ANDROID CHROME: Conserva un riferimento globale all'utterance per evitare che 
        // il Garbage Collector lo elimini prima che abbia finito di pronunciare la frase lunga
        window.utterances = [];

        function parla(testo, prioritario=false) {
            // Se l'utente sta usando il microfono, scarta i vecchi messaggi in ritardo
            if (isRecordingOrWaiting && !prioritario) return;

            if ('speechSynthesis' in window) {
                if (prioritario) {
                    window.speechSynthesis.cancel(); // Ferma tutto per la priorità
                }
                const utterance = new SpeechSynthesisUtterance(testo);
                utterance.lang = 'it-IT';
                
                window.utterances.push(utterance);
                utterance.onend = function() {
                    window.utterances = window.utterances.filter(u => u !== utterance);
                };

                window.speechSynthesis.speak(utterance);
            }
        }

        // Sblocca la riproduzione audio sul primo click dell'utente (Autoplay Policy)
        document.getElementById('btnInit').onclick = () => {
            // Un'azione utente è necessaria per sbloccare speechSynthesis
            const u = new SpeechSynthesisUtterance('');
            window.speechSynthesis.speak(u);
            
            document.getElementById('initOverlay').style.display = 'none';

            // ======= POLLING MESSAGGI DAL SISTEMA IN BACKGROUND =======
            pollingInterval = setInterval(async () => {
                if (isRecordingOrWaiting) return; // Silenzio generale mentre si usa il microfono

                try {
                    const response = await fetch('/api/leggi-messaggi');
                    const data = await response.json();
                    
                    // Ri-controlliamo perché potrebbe essere stato premuto il microfono durante il fetch
                    if (isRecordingOrWaiting) return;

                    if (data.taglia_audio) {
                        window.speechSynthesis.cancel();
                    }

                    if (data.messaggi && data.messaggi.length > 0) {
                        data.messaggi.forEach(msg => {
                            parla(msg);
                        });
                    }
                } catch (e) {
                    // ignora eventuali errori di connessione silenziosamente
                }
            }, 1000);
        };

        // ======= FUNZIONALITÀ REGISTRAZIONE VOCALE =======
        let mediaRecorder;
        let audioChunks = [];

        const btnRecord = document.getElementById('btnRecord');
        const btnStop = document.getElementById('btnStop');
        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');

        btnRecord.onclick = async () => {
            isRecordingOrWaiting = true;
            window.speechSynthesis.cancel(); // Silenzio immediato
            fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=> {}); // Svuota anche la coda sul server

            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = event => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    statusDiv.innerText = 'Elaborazione in corso...';
                    
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    audioChunks = [];

                    const formData = new FormData();
                    formData.append('audio', audioBlob, 'registrazione.webm');

                    try {
                        const response = await fetch('/api/chat-vocale', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        resultDiv.style.display = 'block';
                        
                        if (data.errore) {
                            resultDiv.innerHTML = `<p style="color: #ff4757;"><strong>Errore:</strong> ${data.errore}</p>`;
                            statusDiv.innerText = 'Errore.';
                            parla("Si è verificato un errore di elaborazione.");
                        } else {
                            resultDiv.innerHTML = `
                                <p><strong>Hai detto:</strong> "${data.testo_riconosciuto || '...'}"</p>
                                <p><strong>Risposta Bot:</strong> ${data.risposta_dialogflow}</p>
                            `;
                            statusDiv.innerText = 'Risposta ricevuta!';
                            
                            // Leggi ad alta voce la risposta di Dialogflow
                            if(data.risposta_dialogflow) {
                                parla(data.risposta_dialogflow, true); // Priorità alta
                            }
                            
                            // SE L'INTENTO RIGUARDA L'AMBIENTE, INVIA IL COMANDO A PROVAB
                            // (Nota: ora questa logica è gestita anche direttamente dal server Python per evitare problemi di cache del browser)
                            let intentoLower = data.intento_rilevato.toLowerCase();
                            if(intentoLower.includes('ambient') || intentoLower === 'rilevamento') {
                                fetch('/api/imposta-comando', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({comando: 'W_VOICE'})
                                }).catch(console.error);
                            } else if ((intentoLower.includes('disattiva') && intentoLower.includes('rilevamento')) || intentoLower.includes('navig')) {
                                fetch('/api/imposta-comando', {
                                    method: 'POST',
                                    headers: { 'Content-Type': 'application/json' },
                                    body: JSON.stringify({comando: 'S_VOICE'})
                                }).catch(console.error);
                            } else if (intentoLower.includes('terminazione')) {
                                document.getElementById('initOverlay').style.display = 'flex';
                            }

                            isRecordingOrWaiting = false; // Riprende il polling
                        }
                    } catch (error) {
                        statusDiv.innerText = 'Errore di connessione.';
                        parla("Impossibile connettersi al server.", true);
                        isRecordingOrWaiting = false;
                    }
                };

                mediaRecorder.start();
                statusDiv.innerText = '🎙️ In ascolto... (parla ora)';
                btnRecord.style.display = 'none';
                btnStop.style.display = 'inline-block';
            } catch (err) {
                alert('Errore microfono: ' + err.message);
            }
        };

        btnStop.onclick = () => {
            mediaRecorder.stop();
            btnRecord.style.display = 'inline-block';
            btnStop.style.display = 'none';
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """
    Mostra la pagina HTML con l'interfaccia per registrare il microfono.
    """
    return render_template_string(HTML_PAGE)

@app.route('/api/invia-messaggio', methods=['POST'])
def invia_messaggio():
    """
    Riceve un messaggio testuale da ProvaB e lo mette in coda.
    Se ci sono 3 o più messaggi in coda, svuota e tiene solo l'ultimo, ordinando di tagliare l'audio.
    """
    global taglia_audio_flag
    req = request.get_json(silent=True, force=True)
    if req and 'testo' in req:
        if req.get('taglia_audio', False):
            taglia_audio_flag = True
            messaggi_pendenti.clear()

        messaggi_pendenti.append(req['testo'])
        
        if len(messaggi_pendenti) >= 3:
            messaggi_pendenti[:] = [messaggi_pendenti[-1]]
            taglia_audio_flag = True
            
        return jsonify({"status": "ok"})
    return jsonify({"errore": "Testo mancante"}), 400

@app.route('/api/svuota-messaggi', methods=['POST'])
def svuota_messaggi():
    """
    Svuota forzatamente la coda dei messaggi e ferma l'audio.
    """
    global messaggi_pendenti, taglia_audio_flag
    messaggi_pendenti.clear()
    taglia_audio_flag = True
    return jsonify({"status": "ok"})

@app.route('/api/leggi-messaggi', methods=['GET'])
def leggi_messaggi():
    """
    Fornisce alla pagina web i messaggi accumulati in coda e li rimuove.
    """
    global messaggi_pendenti, taglia_audio_flag
    da_tagliare = taglia_audio_flag
    taglia_audio_flag = False

    if messaggi_pendenti:
        da_leggere = messaggi_pendenti.copy()
        messaggi_pendenti.clear()
        return jsonify({"messaggi": da_leggere, "taglia_audio": da_tagliare})
    return jsonify({"messaggi": [], "taglia_audio": da_tagliare})

@app.route('/api/imposta-comando', methods=['POST'])
def imposta_comando():
    """
    Riceve un comando dall'interfaccia web (es. attivazione intento) 
    e lo memorizza affinché ProvaB possa leggerlo.
    """
    global comando_pendente
    req = request.get_json(silent=True, force=True)
    if req and 'comando' in req:
        comando_pendente = req['comando']
        return jsonify({"status": "ok"})
    return jsonify({"errore": "Comando mancante"}), 400

@app.route('/api/leggi-comando', methods=['GET'])
def leggi_comando():
    """
    Restituisce il comando pendente a ProvaB e lo svuota.
    """
    global comando_pendente
    c = comando_pendente
    comando_pendente = None
    return jsonify({"comando": c})


@app.route('/api/chat-vocale', methods=['POST'])
def dialogflow_voice_chat():
    """
    Riceve un file audio, lo invia alle API di Dialogflow,
    e restituisce l'intento rilevato e la risposta testuale.
    """
    # 1. Verifica che un file audio sia stato inviato nella richiesta
    if 'audio' not in request.files:
        return jsonify({"errore": "Nessun file audio inviato. Invia il file nel form-data come 'audio'."}), 400
        
    file_audio = request.files['audio']
    audio_content = file_audio.read()

    try:
        # 2. Inizializza il client di Dialogflow
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(PROJECT_ID, SESSION_ID)

        # 3. Configura le impostazioni per l'audio in input
        # Nota: AUDIO_ENCODING_UNSPECIFIED o LINEAR_16 sono tra i più comuni. 
        # Modifica sample_rate_hertz se il tuo audio ha una frequenza diversa.
        audio_config = dialogflow.InputAudioConfig(
            audio_encoding=dialogflow.AudioEncoding.AUDIO_ENCODING_UNSPECIFIED,
            sample_rate_hertz=48000,  
            language_code=LANGUAGE_CODE
        )
        
        # Crea la query che raggruppa le configurazioni audio
        query_input = dialogflow.QueryInput(audio_config=audio_config)

        # 4. Invia l'audio a Dialogflow e ottieni la risposta
        response = session_client.detect_intent(
            request={
                "session": session,
                "query_input": query_input,
                "input_audio": audio_content,
            }
        )
        
        # 5. Formatta ed estrai le informazioni utili da mandare al front-end
        risultato = response.query_result
        
        testo_capito = risultato.query_text.strip()
        
        # Se non è stato capito nessun testo (es. silenzio)
        if not testo_capito:
            return jsonify({
                "testo_riconosciuto": "",
                "risposta_dialogflow": "Non ho sentito nulla.",
                "intento_rilevato": "",
                "confidenza": 0.0,
                "parametri": {}
            })
            
        intento_nome = risultato.intent.display_name if hasattr(risultato, 'intent') and risultato.intent else ""
        
        # LOG UTILE PER CAPIRE QUAL È IL VERO NOME DELL'INTENTO SU DIALOGFLOW
        print(f"\n[DIALOGFLOW] Testo Capito: '{testo_capito}'")
        print(f"[DIALOGFLOW] Intento Rilevato: '{intento_nome}'\n")

        # --- LOGICA DI COMANDO DIRETTA (LATO SERVER) ---
        # Se riconosciamo l'intento, impostiamo direttamente il comando_pendente per ProvaB
        global comando_pendente
        intento_lower = intento_nome.lower()
        if 'disattiva' in intento_lower and 'rilevamento' in intento_lower:
            print("[SERVER] Imposto automaticamente il comando 'S_VOICE' per disattivare l'ambiente!")
            comando_pendente = 'S_VOICE'
        elif 'navig' in intento_lower:
            print("[SERVER] Imposto automaticamente il comando 'S_VOICE' per tornare alla navigazione!")
            comando_pendente = 'S_VOICE'
        elif 'ambient' in intento_lower or 'rilevamento' in intento_lower:
            print("[SERVER] Imposto automaticamente il comando 'W_VOICE' per attivare l'ambiente!")
            comando_pendente = 'W_VOICE'
        elif 'terminazione' in intento_lower:
            print("[SERVER] Imposto automaticamente il comando 'Q_VOICE' per terminare!")
            comando_pendente = 'Q_VOICE'


        try:
            parametri_estratti = dict(risultato.parameters) if hasattr(risultato, 'parameters') and risultato.parameters else {}
        except Exception:
            parametri_estratti = {}
        
        # Pulisci eventuali comandi di "taglia_audio" accumulatisi in background mentre
        # l'utente parlava con Dialogflow, per evitare che la risposta venga tagliata sùbito
        global messaggi_pendenti, taglia_audio_flag
        messaggi_pendenti.clear()
        taglia_audio_flag = False

        return jsonify({
            "testo_riconosciuto": testo_capito,
            "risposta_dialogflow": risultato.fulfillment_text,
            "intento_rilevato": intento_nome,
            "confidenza": risultato.intent_detection_confidence,
            "parametri": parametri_estratti
        })

    except Exception as e:
        print(f"Errore Dialogflow API: {e}")
        return jsonify({
            "errore": "Errore di comunicazione con Dialogflow",
            "dettagli": str(e)
        }), 500


@app.route('/api/chat-testo', methods=['POST'])
def dialogflow_text_chat():
    """
    Rotta di utilità per testare la connessione con Dialogflow usando solo testo.
    """
    req = request.get_json(silent=True, force=True)
    if not req or 'testo' not in req:
        return jsonify({"errore": "Devi fornire un campo 'testo' in formato JSON."}), 400

    testo = req.get('testo')

    try:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(PROJECT_ID, SESSION_ID)

        # Passiamo il testo invece dell'audio
        text_input = dialogflow.TextInput(text=testo, language_code=LANGUAGE_CODE)
        query_input = dialogflow.QueryInput(text=text_input)

        response = session_client.detect_intent(
            request={
                "session": session,
                "query_input": query_input,
            }
        )

        risultato = response.query_result
        return jsonify({
            "testo_inviato": risultato.query_text,
            "risposta_dialogflow": risultato.fulfillment_text,
            "intento_rilevato": risultato.intent.display_name,
            "parametri": dict(risultato.parameters)
        })

    except Exception as e:
        return jsonify({"errore": str(e)}), 500


# Manteniamo la route Webhook di base del file originale in caso ti serva
@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)  

    try:
        intent_name = req.get('queryResult').get('intent').get('displayName')
    except AttributeError:
        intent_name = ""

    risposta_testo = "Ho ricevuto il messaggio, ma non è un nuovo ordine."

    if intent_name == 'Nuovo_Ordine':
        parameters = req.get('queryResult', {}).get('parameters', {})
        tipo_pizza_raw = parameters.get('tipo_pizza')
        quantita_raw = parameters.get('number')

        if tipo_pizza_raw and quantita_raw:
            try:
                quantita = int(quantita_raw)  
                tipo_pizza = "Pizza bianca" if tipo_pizza_raw == "Pizza rossa" else "Pizza rossa"
                risposta_testo = f"Perfetto! Ordine per {quantita} {tipo_pizza} registrato con successo."
            except Exception as e:
                risposta_testo = "Ho capito l'ordine, ma c'è un errore tecnico nella lettura dei numeri."
        else:
            risposta_testo = "Dati incompleti per effettuare l'ordine (quantità o tipo di pizza)."
            
    return jsonify({
        "fulfillmentMessages": [{"text": {"text": [risposta_testo]}}]
    })


if __name__ == '__main__':    
    print("\n[INFO] Avvio Server Flask + Dialogflow API")
    print("[INFO] Ricordati di impostare il PROJECT_ID e il SERVICE_ACCOUNT_FILE prima dell'uso.")
    print("------------------------------------------------------------------\n")
    app.run(host='0.0.0.0', port=5000)
