import os
from flask import Flask, request, jsonify, render_template_string, make_response
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

# Variabile globale per memorizzare l'ultimo fotogramma ricevuto dalla PWA
ultimo_frame_ricevuto = None


# ==========================================
# INTERFACCIA WEB (HTML + JAVASCRIPT)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>Centro di Controllo Assistenza</title>
    
    <link rel="manifest" href="/manifest.json">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black">
    <meta name="apple-mobile-web-app-title" content="NavAssist">
    
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 600px; margin: 20px auto; text-align: center; background-color: #121212; color: #ffffff; }
        button { padding: 15px 20px; font-size: 18px; margin: 10px; cursor: pointer; border: none; border-radius: 8px; font-weight: bold; transition: opacity 0.2s; width: 90%; max-width: 400px; }
        button:hover { opacity: 0.8; }
        
        #btnRecord { background-color: #ff4757; color: white; }
        #btnRecord:disabled { background-color: #ff475780; cursor: not-allowed; }
        #btnStop { background-color: #2ed573; color: white; display: none; }
        #btnQuit { background-color: #747d8c; color: white; margin-top: 30px; }
        
        #status { font-size: 1.2em; margin: 20px 0; color: #a4b0be; }
        #result { background-color: #1e272e; padding: 20px; border-radius: 10px; text-align: left; margin-top: 20px; display: none; }
        .highlight { color: #2ed573; font-weight: bold; }
        
        hr { border-color: #2f3542; margin: 30px 0; }
    </style>
</head>
<body>
    <div id="initOverlay" style="position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.9); z-index:9999; display:flex; flex-direction:column; justify-content:center; align-items:center;">
        <h2>Benvenuto nell'Assistente</h2>
        <p>Premi il pulsante per abilitare Fotocamera, Microfono e Voce.</p>
        <button id="btnInit" style="background-color: #2ed573; color: white; padding: 20px 40px; font-size: 24px;">Inizia</button>
    </div>

    <h1>Centro di Controllo</h1>
    <p>Interfaccia Telecomando Web (PWA)</p>
    
    <video id="webcam" autoplay playsinline muted style="width: 95%; max-width: 420px; border-radius: 12px; background-color: #000; margin-bottom: 15px; border: 2px solid #2f3542;"></video>
    <canvas id="hiddenCanvas" style="display: none;"></canvas>
    
    <button id="btnRecord">🔴 Parla con l'Assistente</button>
    <button id="btnStop">⬆️ Ferma e Invia</button>
    
    <br>
    <button id="btnQuit">❌ Spegni Sistema</button>
    
    <div id="status">Inizializzazione in corso...</div>
    <div id="result"></div>

    <script>
        let isRecordingOrWaiting = false;
        let pollingInterval = null;
        let mediaRecorder; 
        let audioChunks = [];

        const videoEl = document.getElementById('webcam');
        const canvasEl = document.getElementById('hiddenCanvas');
        const ctx = canvasEl.getContext('2d');

        const btnRecord = document.getElementById('btnRecord');
        const btnStop = document.getElementById('btnStop');
        const btnQuit = document.getElementById('btnQuit'); // Riferimento al nuovo bottone
        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');

        // ======= FUNZIONALITÀ TEXT-TO-SPEECH (Voce) =======
        window.utterances = [];

        function parla(testo, prioritario=false) {
            if (isRecordingOrWaiting && !prioritario) return;
            if ('speechSynthesis' in window) {
                if (prioritario) window.speechSynthesis.cancel();
                const utterance = new SpeechSynthesisUtterance(testo);
                utterance.lang = 'it-IT';
                window.utterances.push(utterance);
                utterance.onend = () => window.utterances = window.utterances.filter(u => u !== utterance);
                window.speechSynthesis.speak(utterance);
            }
        }

        // ======= INIZIALIZZAZIONE UNIFICATA (Fotocamera, Microfono, Voce) =======
        document.getElementById('btnInit').onclick = async () => {
            // Sblocca la voce
            const u = new SpeechSynthesisUtterance('');
            window.speechSynthesis.speak(u);
            document.getElementById('initOverlay').style.display = 'none';

            try {
                // Richiede fotocamera posteriore e microfono
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: true, 
                    video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } } 
                });
                
                // Assegna il flusso video all'interfaccia
                videoEl.srcObject = stream;
                
                // Estrae solo la traccia audio per inviarla a Dialogflow
                const audioStream = new MediaStream(stream.getAudioTracks());
                mediaRecorder = new MediaRecorder(audioStream);
                
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
                        const response = await fetch('/api/chat-vocale', { method: 'POST', body: formData });
                        const data = await response.json();
                        resultDiv.style.display = 'block';
                        
                        if (data.errore) {
                            resultDiv.innerHTML = `<p style="color: #ff4757;"><strong>Errore:</strong> ${data.errore}</p>`;
                            statusDiv.innerText = 'Errore.';
                            parla("Si è verificato un errore di elaborazione.");
                        } else {
                            resultDiv.innerHTML = `<p><strong>Hai detto:</strong> "${data.testo_riconosciuto || '...'}"</p><p><strong>Risposta Bot:</strong> ${data.risposta_dialogflow}</p>`;
                            statusDiv.innerText = 'Risposta ricevuta!';
                            if(data.risposta_dialogflow) parla(data.risposta_dialogflow, true);
                            
                            let intentoLower = data.intento_rilevato.toLowerCase();
                            if(intentoLower.includes('ambient') || intentoLower === 'rilevamento') {
                                fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'W_VOICE'}) }).catch(console.error);
                            } else if ((intentoLower.includes('disattiva') && intentoLower.includes('rilevamento')) || intentoLower.includes('navig')) {
                                fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'S_VOICE'}) }).catch(console.error);
                            } else if (intentoLower.includes('terminazione')) {
                                // Ora anche la terminazione da voce invia il comando Q_VOICE
                                fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'Q_VOICE'}) }).catch(console.error);
                                setTimeout(() => {
                                    document.body.innerHTML = "<h1 style='margin-top: 50px; color: #ff4757;'>Sistema Disattivato</h1><p>Puoi chiudere questa app.</p>";
                                }, 2000);
                            }
                            isRecordingOrWaiting = false;
                        }
                    } catch (error) {
                        statusDiv.innerText = 'Errore di connessione.';
                        parla("Impossibile connettersi al server.", true);
                        isRecordingOrWaiting = false;
                    }
                };

                // LOOP STREAMING FOTOGRAMMI VERSO IL PC (10 FPS)
               setInterval(() => {
                    // Ignoriamo readyState e controlliamo solo se il video ha effettivamente una dimensione
                    if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
                        // Ridimensiona il canvas nascosto alle stesse dimensioni del video
                        canvasEl.width = videoEl.videoWidth;
                        canvasEl.height = videoEl.videoHeight;
                        
                        // "Scatta la foto" disegnando il frame corrente del video sul canvas
                        ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
                        
                        // Converte il canvas in un file JPEG (qualità 50%) e lo invia a Flask
                        canvasEl.toBlob((blob) => {
                            if(!blob) return;
                            const frameFormData = new FormData();
                            frameFormData.append('frame', blob, 'frame.jpg');
                            fetch('/api/invia-frame', { method: 'POST', body: frameFormData }).catch(() => {});
                        }, 'image/jpeg', 0.5); 
                    }
                }, 100);

                statusDiv.innerText = 'Pronto per registrare.';

            } catch (err) {
                alert('Errore di accesso a microfono/telecamera: ' + err.message);
                statusDiv.innerText = 'Permessi negati.';
            }

            // Polling Messaggi Vocali (dal modulo Python verso il telefono)
            pollingInterval = setInterval(async () => {
                if (isRecordingOrWaiting) return;
                try {
                    const response = await fetch('/api/leggi-messaggi');
                    const data = await response.json();
                    if (isRecordingOrWaiting) return;
                    if (data.taglia_audio) window.speechSynthesis.cancel();
                    if (data.messaggi && data.messaggi.length > 0) {
                        data.messaggi.forEach(msg => parla(msg));
                    }
                } catch (e) {}
            }, 1000);
        };

        // ======= PULSANTI DI CONTROLLO =======
        btnRecord.onclick = () => {
            if (!mediaRecorder) return alert("Dispositivo non pronto. Assicurati di aver dato i permessi.");
            isRecordingOrWaiting = true;
            window.speechSynthesis.cancel();
            fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=>{});

            mediaRecorder.start();
            statusDiv.innerText = '🎙️ In ascolto... (parla ora)';
            btnRecord.style.display = 'none';
            btnStop.style.display = 'inline-block';
        };

        btnStop.onclick = () => {
            mediaRecorder.stop();
            btnRecord.style.display = 'inline-block';
            btnStop.style.display = 'none';
        };

        // ======= PULSANTE DI SPEGNIMENTO =======
        btnQuit.onclick = async () => {
            if (confirm("Vuoi davvero spegnere l'assistente e chiudere tutto?")) {
                statusDiv.innerText = 'Spegnimento in corso...';
                parla("Spegnimento del sistema in corso. A presto.", true);
                
                try {
                    // Invia il comando di "Quit" al server
                    await fetch('/api/imposta-comando', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({comando: 'Q_VOICE'})
                    });
                    
                    // Svuota la pagina per far capire all'utente che è tutto spento
                    setTimeout(() => {
                        document.body.innerHTML = "<h1 style='margin-top: 50px; color: #ff4757;'>Sistema Disattivato</h1><p>Puoi chiudere questa app.</p>";
                    }, 2000); // Aspetta 2 secondi per far finire la frase vocale
                    
                } catch(e) {
                    alert("Errore durante lo spegnimento.");
                }
            }
        };
    </script>
    <script>
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', () => {
                navigator.serviceWorker.register('/sw.js')
                    .then(reg => console.log('Service Worker registrato!'))
                    .catch(err => console.error('Errore Service Worker:', err));
            });
        }
    </script>
</body>
</html>
"""

# ==========================================
# ROTTE PWA (Progressive Web App)
# ==========================================

@app.route('/manifest.json')
def manifest():
    """Fornisce il file di configurazione della PWA al browser."""
    manifest_data = {
        "name": "Assistente di Navigazione",
        "short_name": "NavAssist",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#121212",
        "theme_color": "#2ed573",
        "icons": [
            {
                "src": "https://cdn-icons-png.flaticon.com/512/1086/1086103.png", 
                "sizes": "512x512",
                "type": "image/png"
            }
        ]
    }
    return jsonify(manifest_data)

@app.route('/sw.js')
def service_worker():
    """Fornisce il Service Worker necessario per l'installazione PWA."""
    sw_js = """
    self.addEventListener('install', (e) => {
        console.log('[Service Worker] Installato');
    });
    self.addEventListener('fetch', (e) => {
        e.respondWith(fetch(e.request));
    });
    """
    response = make_response(sw_js)
    response.headers['Content-Type'] = 'application/javascript'
    return response

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# ==========================================
# ROTTE GESTIONE FLUSSO VIDEO
# ==========================================

@app.route('/api/invia-frame', methods=['POST'])
def invia_frame():
    """Riceve un singolo fotogramma JPEG dalla PWA e lo memorizza in memoria."""
    global ultimo_frame_ricevuto
    if 'frame' in request.files:
        ultimo_frame_ricevuto = request.files['frame'].read()
        return jsonify({"status": "ok"})
    return jsonify({"errore": "Frame mancante"}), 400

@app.route('/api/leggi-frame', methods=['GET'])
def leggi_frame():
    """Fornisce l'ultimo fotogramma memorizzato al modulo di navigazione OpenCV."""
    global ultimo_frame_ricevuto
    if ultimo_frame_ricevuto is not None:
        response = make_response(ultimo_frame_ricevuto)
        response.headers['Content-Type'] = 'image/jpeg'
        return response
    return jsonify({"errore": "Nessun frame disponibile"}), 404

# ==========================================
# ROTTE COMUNICAZIONE E AUDIO
# ==========================================

@app.route('/api/invia-messaggio', methods=['POST'])
def invia_messaggio():
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
    global messaggi_pendenti, taglia_audio_flag
    messaggi_pendenti.clear()
    taglia_audio_flag = True
    return jsonify({"status": "ok"})

@app.route('/api/leggi-messaggi', methods=['GET'])
def leggi_messaggi():
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
    global comando_pendente
    req = request.get_json(silent=True, force=True)
    if req and 'comando' in req:
        comando_pendente = req['comando']
        return jsonify({"status": "ok"})
    return jsonify({"errore": "Comando mancante"}), 400

@app.route('/api/leggi-comando', methods=['GET'])
def leggi_comando():
    global comando_pendente
    c = comando_pendente
    comando_pendente = None
    return jsonify({"comando": c})

@app.route('/api/chat-vocale', methods=['POST'])
def dialogflow_voice_chat():
    if 'audio' not in request.files:
        return jsonify({"errore": "Nessun file audio inviato. Invia il file nel form-data come 'audio'."}), 400
        
    file_audio = request.files['audio']
    audio_content = file_audio.read()

    try:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(PROJECT_ID, SESSION_ID)

        audio_config = dialogflow.InputAudioConfig(
            audio_encoding=dialogflow.AudioEncoding.AUDIO_ENCODING_UNSPECIFIED,
            sample_rate_hertz=48000,  
            language_code=LANGUAGE_CODE
        )
        
        query_input = dialogflow.QueryInput(audio_config=audio_config)

        response = session_client.detect_intent(
            request={
                "session": session,
                "query_input": query_input,
                "input_audio": audio_content,
            }
        )
        
        risultato = response.query_result
        testo_capito = risultato.query_text.strip()
        
        if not testo_capito:
            return jsonify({
                "testo_riconosciuto": "",
                "risposta_dialogflow": "Non ho sentito nulla.",
                "intento_rilevato": "",
                "confidenza": 0.0,
                "parametri": {}
            })
            
        intento_nome = risultato.intent.display_name if hasattr(risultato, 'intent') and risultato.intent else ""
        
        print(f"\n[DIALOGFLOW] Testo Capito: '{testo_capito}'")
        print(f"[DIALOGFLOW] Intento Rilevato: '{intento_nome}'\n")

        global comando_pendente
        intento_lower = intento_nome.lower()
        if 'disattiva' in intento_lower and 'rilevamento' in intento_lower:
            comando_pendente = 'S_VOICE'
        elif 'navig' in intento_lower:
            comando_pendente = 'S_VOICE'
        elif 'ambient' in intento_lower or 'rilevamento' in intento_lower:
            comando_pendente = 'W_VOICE'
        elif 'terminazione' in intento_lower:
            comando_pendente = 'Q_VOICE'

        try:
            parametri_estratti = dict(risultato.parameters) if hasattr(risultato, 'parameters') and risultato.parameters else {}
        except Exception:
            parametri_estratti = {}
        
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
    req = request.get_json(silent=True, force=True)
    if not req or 'testo' not in req:
        return jsonify({"errore": "Devi fornire un campo 'testo' in formato JSON."}), 400

    testo = req.get('testo')

    try:
        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(PROJECT_ID, SESSION_ID)

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


if __name__ == '__main__':    
    print("\n[INFO] Avvio Server Flask + Dialogflow API + Video Stream PWA")
    print("[INFO] Ricordati di impostare il PROJECT_ID e il SERVICE_ACCOUNT_FILE prima dell'uso.")
    print("------------------------------------------------------------------\n")
    app.run(host='0.0.0.0', port=5000)