"""Offline speech I/O helpers for the Linux AI PC.

The implementation is intentionally lightweight:
- Faster-Whisper handles offline speech-to-text.
- Piper is called as an external process for fast local TTS.
- Push-to-talk uses the space key via ``pynput`` so the driver decides exactly
  when the engineer should listen.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - optional runtime dependency
    sd = None

try:
    from faster_whisper import WhisperModel
except ImportError:  # pragma: no cover - optional runtime dependency
    WhisperModel = None

try:
    from pynput import keyboard
except ImportError:  # pragma: no cover - optional runtime dependency
    keyboard = None


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"


class PushToTalkRecorder:
    """Records microphone input while the space bar is held down."""

    def __init__(self, audio_config: AudioConfig | None = None) -> None:
        self.audio_config = audio_config or AudioConfig()
        self._frames: list[np.ndarray] = []
        self._recording = threading.Event()
        self._released = threading.Event()
        if np is None or sd is None or keyboard is None:
            raise RuntimeError("Install numpy, sounddevice and pynput for push-to-talk recording")

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if self._recording.is_set():
            self._frames.append(indata.copy())

    def record_once(self) -> bytes:
        self._frames.clear()
        self._released.clear()

        def on_press(key: object) -> None:
            if key == keyboard.Key.space:
                self._recording.set()

        def on_release(key: object) -> bool | None:
            if key == keyboard.Key.space and self._recording.is_set():
                self._recording.clear()
                self._released.set()
                return False
            return None

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            with sd.InputStream(
                samplerate=self.audio_config.sample_rate,
                channels=self.audio_config.channels,
                dtype=self.audio_config.dtype,
                callback=self._audio_callback,
            ):
                print("Hold SPACE to talk to your engineer...")
                while not self._recording.is_set():
                    time.sleep(0.01)
                print("Recording...")
                self._released.wait()
            listener.join()
        print("Recording finished.")
        return self._to_wav_bytes()

    def _to_wav_bytes(self) -> bytes:
        if not self._frames:
            return b""
        audio = np.concatenate(self._frames, axis=0)
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.audio_config.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_config.sample_rate)
            wav_file.writeframes(pcm.tobytes())
        return buffer.getvalue()


class FasterWhisperTranscriber:
    """Wraps Faster-Whisper with a short-form, low-latency configuration."""

    def __init__(self, model_size: str = "small", device: str = "auto", compute_type: str = "int8") -> None:
        if WhisperModel is None:
            raise RuntimeError("Install faster-whisper to enable speech-to-text")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        segments, _info = self.model.transcribe(io.BytesIO(wav_bytes), vad_filter=True, beam_size=1)
        return " ".join(segment.text.strip() for segment in segments).strip()


class PiperSpeaker:
    """Calls Piper and optionally pipes the audio through ffmpeg/sox for radio FX."""

    def __init__(self, model_path: str, output_device: Optional[str] = None, radio_effect: Optional[str] = None) -> None:
        if shutil.which("piper") is None:
            raise RuntimeError("Install Piper and ensure the 'piper' binary is on PATH")
        self.model_path = Path(model_path)
        self.output_device = output_device
        self.radio_effect = radio_effect

    def speak(self, text: str) -> None:
        if not text:
            return
        command = ["piper", "--model", str(self.model_path), "--output_raw"]
        piper_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert piper_process.stdin is not None
        assert piper_process.stdout is not None
        piper_process.stdin.write(text.encode("utf-8"))
        piper_process.stdin.close()

        if self.radio_effect:
            if shutil.which("ffplay") is None:
                raise RuntimeError("Install ffmpeg to use the optional radio effect")
            ffplay_command = [
                "ffplay",
                "-autoexit",
                "-nodisp",
                "-f",
                "s16le",
                "-ar",
                "22050",
                "-ac",
                "1",
                "-af",
                self.radio_effect,
                "-",
            ]
            subprocess.run(ffplay_command, stdin=piper_process.stdout, check=True)
        else:
            aplay_command = ["aplay", "-r", "22050", "-f", "S16_LE", "-c", "1"]
            if self.output_device:
                aplay_command.extend(["-D", self.output_device])
            subprocess.run(aplay_command, stdin=piper_process.stdout, check=True)
