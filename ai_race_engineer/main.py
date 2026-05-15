"""Example entry point that connects telemetry, speech, and Ollama."""

from __future__ import annotations

import argparse
import threading
from typing import Optional

from .ollama_client import OllamaRaceEngineer
from .telemetry import F125TelemetryAdapter, TelemetryStateStore, UDPReceiver
from .voice import FasterWhisperTranscriber, PiperSpeaker, PushToTalkRecorder


def run(
    ollama_model: str = "llama3",
    udp_host: str = "0.0.0.0",
    udp_port: int = 20777,
    piper_model_path: Optional[str] = None,
    radio_effect: Optional[str] = None,
) -> None:
    state = TelemetryStateStore()
    receiver = UDPReceiver(F125TelemetryAdapter(), state, host=udp_host, port=udp_port)
    receiver_thread = threading.Thread(target=receiver.start_forever, daemon=True)
    receiver_thread.start()

    transcriber = FasterWhisperTranscriber(model_size="small")
    recorder = PushToTalkRecorder()
    speaker = PiperSpeaker(piper_model_path=piper_model_path, radio_effect=radio_effect) if piper_model_path else None
    engineer = OllamaRaceEngineer(model=ollama_model)

    print(f"Listening for F1 25 telemetry on {udp_host}:{udp_port} ...")
    while True:
        wav_bytes = recorder.record_once()
        question = transcriber.transcribe_wav_bytes(wav_bytes)
        if not question:
            print("No speech detected, waiting for the next push-to-talk.")
            continue
        telemetry = state.snapshot()
        answer = engineer.answer(question, telemetry)
        print(f"Driver: {question}")
        print(f"Engineer: {answer}")
        if speaker is not None:
            speaker.speak(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lean F1 25 AI race engineer")
    parser.add_argument("--model", default="llama3", help="Ollama model name, e.g. llama3 or phi3")
    parser.add_argument("--udp-host", default="0.0.0.0", help="UDP bind address on the Linux AI PC")
    parser.add_argument("--udp-port", type=int, default=20777, help="F1 25 UDP telemetry port")
    parser.add_argument("--piper-model", default=None, help="Path to the Piper ONNX voice model")
    parser.add_argument(
        "--radio-effect",
        default="highpass=f=300,lowpass=f=3400,acompressor=threshold=-18dB:ratio=3,volume=1.7",
        help="Optional ffmpeg audio filter chain for radio-style voice output",
    )
    args = parser.parse_args()
    run(
        ollama_model=args.model,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        piper_model_path=args.piper_model,
        radio_effect=args.radio_effect if args.piper_model else None,
    )


if __name__ == "__main__":
    main()
