import os
import subprocess
import traceback
from flask import Flask, request, jsonify, render_template_string, make_response
from google.cloud import dialogflow

app = Flask(__name__)

# ==========================================
# CONFIGURAZIONE CONNESSIONE DIALOGFLOW
# ==========================================
PROJECT_ID = "trackbuddy-tpbx"  
SERVICE_ACCOUNT_FILE = "chiave_dialogFlow.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE

# ID sessione fittizio 
SESSION_ID = "sessione-vocale-test-123"
LANGUAGE_CODE = "it-IT"

messaggi_pendenti = []
taglia_audio_flag = False

# Comando pendente in arrivo dall'interfaccia web
comando_pendente = None

# Variabile globale per memorizzare l'ultimo fotogramma ricevuto dalla PWA
ultimo_frame_ricevuto = None


# ==========================================
# CONVERSIONE AUDIO UNIVERSALE (ffmpeg via pipe)
# ==========================================
def converti_audio_in_linear16(audio_bytes):
    """
    Converte qualsiasi formato audio ricevuto dal browser (webm/opus su Android/Chrome,
    mp4/aac su iOS Safari, ecc.) in un formato fisso LINEAR16 PCM mono a 16kHz,
    indipendente dal dispositivo/browser di origine.

    La conversione avviene interamente in memoria tramite pipe stdin/stdout di ffmpeg,
    senza scrittura su disco, per minimizzare la latenza introdotta.
    """
    comando = [
        "ffmpeg",
        "-i", "pipe:0",       # input da stdin, formato auto-rilevato da ffmpeg
        "-ar", "16000",       # sample rate richiesto da Dialogflow
        "-ac", "1",           # mono
        "-f", "wav",          # contenitore WAV (header + LINEAR16 PCM)
        "-acodec", "pcm_s16le",
        "pipe:1"              # output su stdout
    ]
    processo = subprocess.run(
        comando,
        input=audio_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    if processo.returncode != 0:
        raise RuntimeError(f"Conversione ffmpeg fallita: {processo.stderr.decode(errors='ignore')}")
    return processo.stdout


# ==========================================
# INTERFACCIA PWA
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="it">
<head>
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrackBuddy - Centro di Controllo</title>
    
    <link rel="manifest" href="/manifest.json">
    
    <link rel="icon" href="/static/Icon.png?v=3" type="image/png">
    <link rel="apple-touch-icon" href="/static/Icon.png?v=3">
    
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="TrackBuddy">
    
    <style>
        :root {
            --bg-light: #f8f9fa; 
            --text-dark: #2d3436; 
            --focus-ring: #0984e3; 
        }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 40rem;
            margin: 1.5rem auto; 
            text-align: center; 
            background-color: var(--bg-light); 
            color: var(--text-dark); 
            font-size: 1.125rem; 
            line-height: 1.6;
        }
        
        button { 
            padding: 1rem 1.5rem; 
            font-size: 1.25rem; 
            margin: 0.75rem; 
            cursor: pointer; 
            border: 2px solid transparent; 
            border-radius: 0.5rem; 
            font-weight: bold; 
            transition: opacity 0.2s; 
            width: 90%; 
            max-width: 25rem; 
        }
        
        /* Outline per navigazione da tastiera / accessibilità motoria */
        button:focus-visible {
            outline: 4px solid var(--focus-ring);
            outline-offset: 4px;
        }
        
        button:hover { opacity: 0.8; }
        
        /* Colori ricalibrati per contrasto con testo bianco (Livello AAA) */
        #btnRecord { background-color: #d63031; color: white; border: 2px solid #d63031; }
        #btnRecord:disabled { background-color: #ff7675; cursor: not-allowed; border-color: transparent; color: #fdfdfd; }
        #btnStop { background-color: #00b894; color: white; border: 2px solid #00b894; display: none; } 
        #btnQuit { background-color: #636e72; color: white; margin-top: 2rem; border: 2px solid #636e72; }
        
        #status { font-size: 1.25rem; margin: 1.5rem 0; color: #2d3436; font-weight: 700; }
        
        #result { 
            background-color: #ffffff; 
            padding: 1.5rem; 
            border-radius: 0.75rem; 
            text-align: left; 
            margin-top: 1.5rem; 
            display: none; 
            border: 2px solid #b2bec3; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        .highlight { color: #00b894; font-weight: bold; }
        
        #webcam { 
            width: 95%; 
            max-width: 26rem; 
            border-radius: 0.75rem; 
            background-color: #000; 
            margin-bottom: 1rem; 
            border: 3px solid #636e72; 
        }
        
        hr { border-color: #dfe6e9; margin: 2rem 0; }
    </style>
</head>
<body>
    <div id="initOverlay" role="dialog" aria-labelledby="overlayTitle" aria-describedby="overlayDesc" style="position: fixed; top:0; left:0; width:100%; height:100%; background:rgba(248, 249, 250, 0.98); z-index:9999; display:flex; flex-direction:column; justify-content:center; align-items:center;">
        <h2 id="overlayTitle" style="color: #2d3436;">Benvenuto nell'Assistente</h2>
        <p id="overlayDesc" style="font-size: 1.2rem; padding: 0 2rem; color: #2d3436;">Premi il pulsante per abilitare Fotocamera, Microfono e Voce.</p>
        <button id="btnInit" style="background-color: #00b894; color: white; padding: 1.5rem 2.5rem; font-size: 1.5rem; border: none; font-weight: 800;">Inizia</button>
    </div>

    <main>
        <header>
            <h1>Centro di Controllo</h1>
            <p>Interfaccia Telecomando Web (PWA)</p>
        </header>
        
        <section aria-label="Area fotocamera in tempo reale">
            <video id="webcam" aria-label="Ripresa della fotocamera posteriore in tempo reale" autoplay playsinline muted></video>
            <div id="debugLog" style="font-size:0.75rem; word-break:break-all; background:#dfe6e9; padding:8px; text-align:left; border-radius:8px; margin:8px 0;"></div>
            <canvas id="hiddenCanvas" style="display: none;" aria-hidden="true"></canvas>
        </section>
        
        <section aria-label="Controlli principali">
            <button id="btnRecord" aria-label="Avvia la registrazione vocale">🔴 Parla con l'Assistente</button>
            <button id="btnStop" aria-label="Ferma la registrazione e invia la richiesta">⬆️ Ferma e Invia</button>
            <br>
            <button id="btnQuit" aria-label="Spegni definitivamente il sistema">❌ Spegni Sistema</button>
        </section>
        
        <section aria-label="Stato del sistema">
            <div id="status" aria-live="polite" aria-atomic="true">Inizializzazione in corso...</div>
            
            <div id="result" aria-live="assertive"></div>
        </section>
    </main>

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
        const btnQuit = document.getElementById('btnQuit');
        const statusDiv = document.getElementById('status');
        const resultDiv = document.getElementById('result');
        const debugLogElement = document.getElementById('debugLog');

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

        // ======= INIZIALIZZAZIONE UNIFICATA =======
        document.getElementById('btnInit').onclick = async () => {
            const u = new SpeechSynthesisUtterance('');
            window.speechSynthesis.speak(u);
            document.getElementById('initOverlay').style.display = 'none';
            try {
                let tempStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
                const dispositivi = await navigator.mediaDevices.enumerateDevices();
                const fotocamere = dispositivi.filter(device => device.kind === 'videoinput');
                tempStream.getTracks().forEach(track => track.stop());

                let idFotocameraTarget = null;
                if (fotocamere.length > 2) {
                    idFotocameraTarget = fotocamere[2].deviceId;
                } else if (fotocamere.length > 1) {
                    idFotocameraTarget = fotocamere[1].deviceId;
                } else if (fotocamere.length > 0) {
                    idFotocameraTarget = fotocamere[0].deviceId;
                }

                const vincoliFlusso = {
                    audio: true,
                    video: idFotocameraTarget ? { deviceId: { exact: idFotocameraTarget }, width: { ideal: 640 }, height: { ideal: 480 } } : { facingMode: "environment" }
                };
                const stream = await navigator.mediaDevices.getUserMedia(vincoliFlusso);
                videoEl.srcObject = stream;
                
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
                            resultDiv.innerHTML = `<p style="color: #d63031;"><strong>Errore:</strong> ${data.errore}</p>`;
                            statusDiv.innerText = 'Errore.';
                            
                            parla("Si è verificato un errore. Riprenderò la navigazione assistita, attendere nuove indicazioni.", true);
                            
                            setTimeout(() => {
                                isRecordingOrWaiting = false; 
                                statusDiv.innerText = 'Pronto per registrare.'; 
                                resultDiv.style.display = 'none'; 
                                // 3. Svuota la coda dei messaggi vecchi di Python per sbloccare la navigazione!
                                fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=>{});
                                fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'S_VOICE'}) }).catch(console.error);
                            }, 5500);
                            
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
                                // 1. Disattiva subito il polling e cancella la coda audio residua del browser
                                if (pollingInterval) clearInterval(pollingInterval);
                                window.speechSynthesis.cancel();
                                
                                fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=>{});
                                fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'Q_VOICE'}) }).catch(console.error);
                                
                                setTimeout(() => {
                                    document.body.innerHTML = "<h1 style='margin-top: 50px; color: #d63031;'>Sistema Disattivato</h1><p>Puoi chiudere questa app.</p>";
                                }, 2000);
                            }
                            isRecordingOrWaiting = false;
                        }
                    } catch (error) {
                        statusDiv.innerText = 'Errore di connessione.';
                        
                        parla("Si è verificato un errore di connessione. Riprenderò la navigazione assistita, attendere nuove indicazioni.", true);
                        
                        setTimeout(() => {
                            isRecordingOrWaiting = false;
                            statusDiv.innerText = 'Pronto per registrare.'; 
                            resultDiv.style.display = 'none'; 
                            fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=>{});
                            fetch('/api/imposta-comando', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({comando: 'S_VOICE'}) }).catch(console.error);
                        }, 5500);
                    }
                };
               setInterval(() => {
                    if (videoEl.videoWidth > 0 && videoEl.videoHeight > 0) {
                        canvasEl.width = videoEl.videoWidth;
                        canvasEl.height = videoEl.videoHeight;
                        ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
                        
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
                
                if (pollingInterval) clearInterval(pollingInterval);
                
                window.speechSynthesis.cancel();
                
                parla("Spegnimento del sistema in corso. A presto.", true);
                
                try {
                    await fetch('/api/svuota-messaggi', {method: 'POST'}).catch(()=>{});
                    
                    await fetch('/api/imposta-comando', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({comando: 'Q_VOICE'})
                    });
                    
                    setTimeout(() => {
                        document.body.innerHTML = "<h1 style='margin-top: 50px; color: #d63031;'>Sistema Disattivato</h1><p>Puoi chiudere questa app.</p>";
                    }, 2000);
                    
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
        "name": "TrackBuddy - Assistente di Navigazione",
        "short_name": "TrackBuddy",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#f8f9fa",
        "theme_color": "#0984e3",
        "icons": [
            {
                "src": "/static/Icon.png?v=3", 
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/static/Icon.png?v=3", 
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
    audio_grezzo = file_audio.read()

    try:
        # Conversione universale: indipendentemente dal formato inviato dal browser
        # (webm/opus su Chrome/Android, mp4/aac su Safari/iOS), l'audio arriva qui
        # sempre come LINEAR16 PCM 16kHz mono.
        audio_content = converti_audio_in_linear16(audio_grezzo)

        session_client = dialogflow.SessionsClient()
        session = session_client.session_path(PROJECT_ID, SESSION_ID)

        audio_config = dialogflow.InputAudioConfig(
            audio_encoding=dialogflow.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=16000,
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
        print("--- STACK TRACE COMPLETO (diagnostica temporanea) ---")
        traceback.print_exc()
        print("-------------------------------------------------------")
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