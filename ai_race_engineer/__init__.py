"""Lean AI race engineer reference implementation for F1 25 telemetry."""

from .ollama_client import OllamaRaceEngineer
from .telemetry import F125TelemetryAdapter, TelemetryStateStore, UDPReceiver


def run(*args, **kwargs):
    from .main import run as _run

    return _run(*args, **kwargs)


__all__ = [
    "F125TelemetryAdapter",
    "OllamaRaceEngineer",
    "TelemetryStateStore",
    "UDPReceiver",
    "run",
]
