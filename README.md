# mr-telemetry

Ein schlankes, modular erweiterbares Referenzprojekt für einen lokalen **AI Race Engineer** rund um **F1 25**, **Ollama** und **Offline-Sprachverarbeitung**.

## 1. Architektur-Übersicht

### Kommunikationsfluss

- **Gaming-PC (Windows)**
  - F1 25 sendet UDP-Telemetrie direkt an den Linux-KI-PC.
  - Standard-Ziel: `UDP 20777` auf der IP des Linux-PCs.
- **KI-PC (Linux / Ubuntu)**
  - `ai_race_engineer.telemetry.UDPReceiver` lauscht auf `0.0.0.0:20777` und aktualisiert ein thread-sicheres Telemetrie-Dictionary.
  - `Faster-Whisper` verarbeitet Push-to-Talk-Audio lokal.
  - `Ollama` läuft lokal unter `http://localhost:11434/v1/chat/completions`.
  - `Piper` spricht die Antwort lokal aus; optional legt `ffmpeg`/`SoX` einen Funk-Effekt darüber.

### Modulgrenzen

- `/ai_race_engineer/telemetry.py` – UDP-Empfang, Parsing der F1-25-Pakete, Race-State.
- `/ai_race_engineer/voice.py` – Push-to-Talk, Faster-Whisper, Piper-TTS.
- `/ai_race_engineer/ollama_client.py` – extrem kompakte Ollama-Integration.
- `/ai_race_engineer/main.py` – Orchestrierung des Gesamtsystems.
- `/tests/` – fokussierte `unittest`-Tests für Parser und Ollama-Payload.

### Erweiterbarkeit für LMU / NASCAR / andere Sims

Die Erweiterung ist bewusst über eine kleine `TelemetryAdapter`-Schnittstelle vorbereitet. Für andere Spiele musst du nur einen neuen Adapter schreiben, der eingehende Telemetrie-Pakete in dieselbe Snapshot-Struktur übersetzt. STT, TTS und Ollama bleiben unverändert.

## 2. Telemetrie-Parser & UDP-Empfang

### Wichtiger Architekturhinweis

Für die im Problem genannten Werte werden in der Praxis **mehrere F1-Pakete** benötigt:

- `PacketLapData` → Runde, Position, Gaps, Feldübersicht
- `PacketCarTelemetryData` → Speed, Gas, Bremse, DRS, Reifendrücke/Temperaturen
- `PacketCarStatusData` → `m_fuelInTank`, `m_ersStoreEnergy`
- `PacketCarDamageData` → `m_tyreWear`

Darum dekodiert das Beispiel bewusst **alle vier** Pakete, obwohl dein Kern-Fokus auf `PacketCarTelemetryData` und `PacketLapData` liegt.

### Beispiel: Live-UDP-Empfang und Parsing

```python
from ai_race_engineer.telemetry import F125TelemetryAdapter, TelemetryStateStore, UDPReceiver

state = TelemetryStateStore()
receiver = UDPReceiver(F125TelemetryAdapter(), state, host="0.0.0.0", port=20777)
receiver.start_forever()
```

### Snapshot-Struktur für die KI

Der Parser hält einen global nutzbaren Snapshot aktuell, zum Beispiel:

```python
{
  "player": {
    "current_lap": 14,
    "position": 5,
    "speed_kph": 308,
    "fuel_in_tank_l": 17.2,
    "fuel_remaining_laps": 10.4,
    "ers_store_energy_j": 2450000.0,
    "tyre_wear_pct": {"FL": 28.0, "FR": 30.0, "RL": 22.0, "RR": 23.0},
    "lap_gain_seconds": 0.184
  },
  "race": {
    "gap_to_car_ahead_s": 0.912,
    "gap_to_leader_s": 7.421,
    "car_ahead": {"position": 4, "distance_m": 18.4},
    "car_behind": {"position": 6, "distance_m": 11.8}
  }
}
```

Damit kann die KI nicht nur Reifen/Benzin/ERS beantworten, sondern auch **Abstände, Pace-Gewinn pro Runde und situatives Racecraft-Feedback** geben.

## 3. Sprach-Steuerung (Linux-PC)

### STT: Faster-Whisper

- Offline, schnell und robust.
- Im Beispiel wird **Push-to-Talk via Leertaste** genutzt.
- Das Mikrofon hängt direkt am Linux-PC.

Wichtige Klasse:

```python
from ai_race_engineer.voice import FasterWhisperTranscriber, PushToTalkRecorder

recorder = PushToTalkRecorder()
transcriber = FasterWhisperTranscriber(model_size="small")
wav_bytes = recorder.record_once()
text = transcriber.transcribe_wav_bytes(wav_bytes)
```

### TTS: Piper

```python
from ai_race_engineer.voice import PiperSpeaker

speaker = PiperSpeaker(
    piper_model_path="/opt/piper/en_GB-alan-medium.onnx",
    radio_effect="highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=3,volume=1.7",
)
speaker.speak("Box this lap, front tyres are dropping away.")
```

### Funk-Effekt / Boxenfunk-Sound

Für einen einfachen Boxenfunk-Charakter reicht oft schon ein enger Sprachbandpass plus leichte Kompression:

- **ffmpeg**-Filterkette (bereits im Code vorbereitet):
  - `highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=3,volume=1.7`
- Alternativ mit **SoX**:
  - `sox input.wav output.wav highpass 300 lowpass 3400 compand 0.02,0.20 6:-70,-60,-20 -5 -90 0.2`

## 4. Ollama-Integration & System Prompt

Die Funktion `OllamaRaceEngineer.answer(...)` sendet die aktuelle Telemetrie zusammen mit der Fahrerfrage als JSON an die lokale Ollama-API:

```python
from ai_race_engineer.ollama_client import OllamaRaceEngineer

engineer = OllamaRaceEngineer(model="llama3")
reply = engineer.answer(
    "Wie sehen die Vorderreifen aus?",
    telemetry_snapshot,
)
```

### Verwendeter System-Prompt

```text
Du bist der Renningenieur eines F1-Fahrers.
Antworte immer professionell, ruhig und entscheidungsorientiert wie echter Boxenfunk.
Nutze ausschließlich die mitgelieferte Telemetrie und erfinde keine Werte.
Baue die wichtigsten Zahlen aktiv in die Antwort ein, zum Beispiel Reifenverschleiß, Benzin, ERS, Gap oder Pace.
Antworte maximal in 1 bis 2 kurzen Sätzen mit klarer Handlungsempfehlung.
Wenn Telemetrie fehlt, sage knapp, was fehlt, statt zu raten.
Priorisiere Sicherheit, Reifenmanagement, Energieeinsatz, Boxenstopp-Fenster, Überholen und Verteidigung.
```

### Warum dieser Prompt funktioniert

- **Boxenfunk-Tonfall** statt Chatbot-Stil.
- **Hartes Satzlimit** für minimale Ablenkung beim Fahren.
- **Zahlenpflicht** sorgt dafür, dass echte Telemetrie in der Antwort landet.
- **Keine Halluzinationen** bei fehlenden Daten.

## 5. Schritt-für-Schritt Einrichtungsanleitung

### Windows (Gaming-PC)

1. In F1 25 die UDP-Telemetrie aktivieren.
2. Als Ziel-IP die Linux-IP eintragen.
3. Port auf `20777` setzen.
4. Sendefrequenz `20Hz` oder `30Hz` wählen.
5. Python ist auf Windows nur nötig, falls du später einen zusätzlichen lokalen Forwarder oder Debug-Tools starten willst.

**Optionales Windows-Paketset für Debug/Replay:**

```bash
py -m pip install --upgrade pip
py -m pip install numpy
```

### Linux (KI-PC)

#### Python-Abhängigkeiten

```bash
python3 -m pip install --upgrade pip
python3 -m pip install numpy sounddevice faster-whisper pynput
```

#### System-Tools

```bash
sudo apt update
sudo apt install -y ffmpeg alsa-utils
```

#### Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3
# alternativ:
ollama pull phi3
```

#### Piper TTS

- Piper installieren
- ONNX-Stimme herunterladen, z. B. `en_GB-alan-medium.onnx`
- Pfad später an `--piper-model` übergeben

### Projekt starten

#### Nur Telemetrie + Sprachdialog + Ollama

```bash
cd ~/mr-telemetry
python3 -m ai_race_engineer --model llama3 --udp-port 20777
```

#### Mit Piper und Funk-Effekt

```bash
cd ~/mr-telemetry
python3 -m ai_race_engineer \
  --model llama3 \
  --udp-port 20777 \
  --piper-model /opt/piper/en_GB-alan-medium.onnx
```

## Produktionsnahe Betriebs-Tipps

- `small` oder `base` für Faster-Whisper ist meistens der beste Latenz-Kompromiss.
- Nutze auf dem Linux-PC möglichst ein dediziertes USB-Headset.
- Halte Ollama lokal und ohne Streaming, damit die Antwort deterministischer bleibt.
- Wenn du später LMU, iRacing oder NASCAR ergänzen willst, behalte dieselbe Snapshot-Struktur für die KI-Ausgabe bei. Genau das macht den Code hier modular und leicht erweiterbar.

## Entwicklung & Tests

```bash
cd ~/mr-telemetry
python3 -m unittest discover -s tests -v
```
