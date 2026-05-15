"""Local Ollama chat client for the AI race engineer."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List

SYSTEM_PROMPT = """Du bist der Renningenieur eines F1-Fahrers.
Antworte immer professionell, ruhig und entscheidungsorientiert wie echter Boxenfunk.
Nutze ausschließlich die mitgelieferte Telemetrie und erfinde keine Werte.
Baue die wichtigsten Zahlen aktiv in die Antwort ein, zum Beispiel Reifenverschleiß, Benzin, ERS, Gap oder Pace.
Antworte maximal in 1 bis 2 kurzen Sätzen mit klarer Handlungsempfehlung.
Wenn Telemetrie fehlt, sage knapp, was fehlt, statt zu raten.
Priorisiere Sicherheit, Reifenmanagement, Energieeinsatz, Boxenstopp-Fenster, Überholen und Verteidigung."""


@dataclass
class OllamaRaceEngineer:
    model: str = "llama3"
    endpoint: str = "http://localhost:11434/v1/chat/completions"
    temperature: float = 0.2
    timeout: int = 20

    def build_messages(self, driver_question: str, telemetry: Dict[str, Any]) -> List[Dict[str, str]]:
        payload = {
            "driver_question": driver_question,
            "telemetry": telemetry,
            "response_style": {
                "language": "de",
                "max_sentences": 2,
                "tone": "professional_race_engineer",
            },
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ]

    def answer(self, driver_question: str, telemetry: Dict[str, Any]) -> str:
        request_body = {
            "model": self.model,
            "messages": self.build_messages(driver_question, telemetry),
            "temperature": self.temperature,
            "stream": False,
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            response_body = json.loads(response.read().decode("utf-8"))
        return response_body["choices"][0]["message"]["content"].strip()
