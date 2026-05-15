"""Telemetry parsing primitives for F1 25 style UDP packets.

The packet layout follows the Codemasters / EA SPORTS F1 telemetry protocol that
has remained stable across recent releases. The parser is isolated behind the
``TelemetryAdapter`` interface so that other sims (LMU, NASCAR, iRacing, ...)
can be added without touching the voice or LLM pipeline.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Protocol

PLAYER_CAR_COUNT = 22


class PacketIds:
    """Known F1 packet ids used by the lightweight parser."""

    LAP_DATA = 2
    CAR_TELEMETRY = 6
    CAR_STATUS = 7
    CAR_DAMAGE = 10


HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")
LAP_DATA_STRUCT = struct.Struct("<IIHBHBHHfffBBBBBBBBBBBBBBBHHB")
CAR_TELEMETRY_STRUCT = struct.Struct("<HfffBbHBBH4H4B4BH4f4B")
CAR_STATUS_STRUCT = struct.Struct("<BBBBBfffHHBBHBBBbfffBfffB")
CAR_DAMAGE_STRUCT = struct.Struct("<4f26B")


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


@dataclass(frozen=True)
class PacketHeader:
    packet_format: int
    game_year: int
    packet_version: int
    packet_id: int
    player_car_index: int

    @classmethod
    def from_buffer(cls, payload: bytes) -> "PacketHeader":
        if len(payload) < HEADER_STRUCT.size:
            raise ValueError("packet shorter than telemetry header")
        unpacked = HEADER_STRUCT.unpack_from(payload)
        return cls(
            packet_format=unpacked[0],
            game_year=unpacked[1],
            packet_version=unpacked[4],
            packet_id=unpacked[5],
            player_car_index=unpacked[10],
        )


class TelemetryAdapter(Protocol):
    """Protocol for game specific telemetry decoders."""

    def parse_packet(self, payload: bytes, state: "TelemetryStateStore") -> None:
        ...


class TelemetryStateStore:
    """Thread-safe store that keeps the latest derived telemetry snapshot."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "session": {
                "game": "F1 25",
                "packet_format": 2025,
                "updated_at": 0.0,
                "lap_history_ms": {},
            },
            "player": {
                "current_lap": 0,
                "position": None,
                "speed_kph": 0,
                "gear": 0,
                "throttle": 0.0,
                "brake": 0.0,
                "drs_active": False,
                "fuel_in_tank_l": 0.0,
                "fuel_remaining_laps": 0.0,
                "ers_store_energy_j": 0.0,
                "tyre_wear_pct": {"RL": 0.0, "RR": 0.0, "FL": 0.0, "FR": 0.0},
                "tyre_pressure_psi": {"RL": 0.0, "RR": 0.0, "FL": 0.0, "FR": 0.0},
                "tyre_surface_temp_c": {"RL": 0, "RR": 0, "FL": 0, "FR": 0},
                "tyre_inner_temp_c": {"RL": 0, "RR": 0, "FL": 0, "FR": 0},
                "last_lap_time_ms": 0,
                "current_lap_time_ms": 0,
                "lap_gain_seconds": 0.0,
            },
            "race": {
                "gap_to_leader_s": 0.0,
                "gap_to_car_ahead_s": 0.0,
                "car_ahead": None,
                "car_behind": None,
                "field": [],
            },
        }

    def update(self, updater: Any) -> None:
        with self._lock:
            updater(self._state)
            self._state["session"]["updated_at"] = time.time()

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._state))


class F125TelemetryAdapter:
    """Parses the core F1 25 telemetry packets needed by the race engineer."""

    tyre_order = ("RL", "RR", "FL", "FR")

    def parse_packet(self, payload: bytes, state: TelemetryStateStore) -> None:
        header = PacketHeader.from_buffer(payload)
        if header.packet_format != 2025:
            return
        if header.packet_id == PacketIds.CAR_TELEMETRY:
            self._parse_car_telemetry(payload, header, state)
        elif header.packet_id == PacketIds.CAR_STATUS:
            self._parse_car_status(payload, header, state)
        elif header.packet_id == PacketIds.CAR_DAMAGE:
            self._parse_car_damage(payload, header, state)
        elif header.packet_id == PacketIds.LAP_DATA:
            self._parse_lap_data(payload, header, state)

    def _player_chunk(self, payload: bytes, struct_obj: struct.Struct, player_car_index: int) -> tuple[Any, ...]:
        offset = HEADER_STRUCT.size + (player_car_index * struct_obj.size)
        end_offset = offset + struct_obj.size
        if len(payload) < end_offset:
            raise ValueError("packet shorter than expected player chunk")
        return struct_obj.unpack_from(payload, offset)

    def _parse_car_telemetry(self, payload: bytes, header: PacketHeader, state: TelemetryStateStore) -> None:
        values = self._player_chunk(payload, CAR_TELEMETRY_STRUCT, header.player_car_index)
        tyres_pressure = {name: _round(values[23 + index], 1) for index, name in enumerate(self.tyre_order)}
        tyres_surface_temp = {name: int(values[14 + index]) for index, name in enumerate(self.tyre_order)}
        tyres_inner_temp = {name: int(values[18 + index]) for index, name in enumerate(self.tyre_order)}

        def updater(current: MutableMapping[str, Any]) -> None:
            player = current["player"]
            player.update(
                {
                    "speed_kph": int(values[0]),
                    "throttle": _round(values[1], 3),
                    "brake": _round(values[3], 3),
                    "gear": int(values[5]),
                    "engine_rpm": int(values[6]),
                    "drs_active": bool(values[7]),
                    "tyre_pressure_psi": tyres_pressure,
                    "tyre_surface_temp_c": tyres_surface_temp,
                    "tyre_inner_temp_c": tyres_inner_temp,
                }
            )

        state.update(updater)

    def _parse_car_status(self, payload: bytes, header: PacketHeader, state: TelemetryStateStore) -> None:
        values = self._player_chunk(payload, CAR_STATUS_STRUCT, header.player_car_index)

        def updater(current: MutableMapping[str, Any]) -> None:
            player = current["player"]
            player.update(
                {
                    "fuel_in_tank_l": _round(values[5], 2),
                    "fuel_capacity_l": _round(values[6], 2),
                    "fuel_remaining_laps": _round(values[7], 2),
                    "drs_allowed": bool(values[11]),
                    "tyres_age_laps": int(values[15]),
                    "engine_power_ice_w": _round(values[17], 2),
                    "engine_power_mguk_w": _round(values[18], 2),
                    "ers_store_energy_j": _round(values[19], 2),
                    "ers_deploy_mode": int(values[20]),
                    "ers_harvested_mguk_j": _round(values[21], 2),
                    "ers_harvested_mguh_j": _round(values[22], 2),
                    "ers_deployed_j": _round(values[23], 2),
                }
            )

        state.update(updater)

    def _parse_car_damage(self, payload: bytes, header: PacketHeader, state: TelemetryStateStore) -> None:
        values = self._player_chunk(payload, CAR_DAMAGE_STRUCT, header.player_car_index)
        tyre_wear = {name: _round(values[index], 1) for index, name in enumerate(self.tyre_order)}

        def updater(current: MutableMapping[str, Any]) -> None:
            player = current["player"]
            player.update(
                {
                    "tyre_wear_pct": tyre_wear,
                    "front_wing_damage_pct": {"left": int(values[8]), "right": int(values[9])},
                    "rear_wing_damage_pct": int(values[10]),
                    "floor_damage_pct": int(values[11]),
                }
            )

        state.update(updater)

    def _parse_lap_data(self, payload: bytes, header: PacketHeader, state: TelemetryStateStore) -> None:
        entries: List[Dict[str, Any]] = []
        expected_size = HEADER_STRUCT.size + (PLAYER_CAR_COUNT * LAP_DATA_STRUCT.size)
        if len(payload) < expected_size:
            raise ValueError("lap data packet shorter than expected field array")

        for index in range(PLAYER_CAR_COUNT):
            offset = HEADER_STRUCT.size + (index * LAP_DATA_STRUCT.size)
            values = LAP_DATA_STRUCT.unpack_from(payload, offset)
            total_distance = values[9] if values[9] > 0 else values[8]
            entries.append(
                {
                    "car_index": index,
                    "last_lap_time_ms": int(values[0]),
                    "current_lap_time_ms": int(values[1]),
                    "delta_to_car_in_front_ms": int(values[6]),
                    "delta_to_leader_ms": int(values[7]),
                    "lap_distance_m": _round(values[8], 2),
                    "total_distance_m": _round(total_distance, 2),
                    "safety_car_delta": _round(values[10], 3),
                    "position": int(values[11]),
                    "current_lap": int(values[12]),
                    "pit_status": int(values[13]),
                    "sector": int(values[15]),
                    "current_lap_invalid": bool(values[16]),
                    "penalties": int(values[17]),
                    "result_status": int(values[24]),
                }
            )

        player_entry = entries[header.player_car_index]
        sorted_field = sorted(
            (entry for entry in entries if entry["result_status"] != 0),
            key=lambda entry: (entry["position"] or 99, -entry["total_distance_m"]),
        )
        ahead = next((entry for entry in sorted_field if entry["position"] == player_entry["position"] - 1), None)
        behind = next((entry for entry in sorted_field if entry["position"] == player_entry["position"] + 1), None)

        def updater(current: MutableMapping[str, Any]) -> None:
            player = current["player"]
            session = current["session"]
            race = current["race"]

            completed_lap = max(player_entry["current_lap"] - 1, 0)
            lap_history = session["lap_history_ms"]
            last_lap_ms = player_entry["last_lap_time_ms"]
            if completed_lap and last_lap_ms:
                lap_history[str(completed_lap)] = last_lap_ms
                previous_lap_ms = lap_history.get(str(completed_lap - 1), last_lap_ms)
                lap_gain_seconds = _round((previous_lap_ms - last_lap_ms) / 1000.0, 3)
            else:
                lap_gain_seconds = 0.0

            player.update(
                {
                    "current_lap": player_entry["current_lap"],
                    "position": player_entry["position"],
                    "last_lap_time_ms": player_entry["last_lap_time_ms"],
                    "current_lap_time_ms": player_entry["current_lap_time_ms"],
                    "lap_distance_m": player_entry["lap_distance_m"],
                    "total_distance_m": player_entry["total_distance_m"],
                    "lap_gain_seconds": lap_gain_seconds,
                }
            )
            race.update(
                {
                    "gap_to_leader_s": _round(player_entry["delta_to_leader_ms"] / 1000.0, 3),
                    "gap_to_car_ahead_s": _round(player_entry["delta_to_car_in_front_ms"] / 1000.0, 3),
                    "car_ahead": {
                        "car_index": ahead["car_index"],
                        "position": ahead["position"],
                        "distance_m": _round(ahead["total_distance_m"] - player_entry["total_distance_m"], 2),
                    }
                    if ahead
                    else None,
                    "car_behind": {
                        "car_index": behind["car_index"],
                        "position": behind["position"],
                        "distance_m": _round(player_entry["total_distance_m"] - behind["total_distance_m"], 2),
                    }
                    if behind
                    else None,
                    "field": [
                        {
                            "car_index": item["car_index"],
                            "position": item["position"],
                            "delta_to_leader_s": _round(item["delta_to_leader_ms"] / 1000.0, 3),
                            "lap_distance_m": item["lap_distance_m"],
                        }
                        for item in sorted_field
                    ],
                }
            )

        state.update(updater)


class UDPReceiver:
    """Blocking UDP listener that continuously feeds a telemetry adapter."""

    def __init__(
        self,
        adapter: TelemetryAdapter,
        state: TelemetryStateStore,
        host: str = "0.0.0.0",
        port: int = 20777,
        buffer_size: int = 4096,
    ) -> None:
        self.adapter = adapter
        self.state = state
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self._socket: socket.socket | None = None
        self._stop_event = threading.Event()

    def start_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.host, self.port))
            self._socket = sock
            while not self._stop_event.is_set():
                payload, _address = sock.recvfrom(self.buffer_size)
                self.adapter.parse_packet(payload, self.state)

    def stop(self) -> None:
        self._stop_event.set()
        if self._socket is not None:
            self._socket.close()


def build_default_snapshot() -> Dict[str, Any]:
    """Convenience helper used by docs and tests."""

    return TelemetryStateStore().snapshot()
