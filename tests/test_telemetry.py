import struct
import unittest

from ai_race_engineer.telemetry import (
    CAR_DAMAGE_STRUCT,
    CAR_STATUS_STRUCT,
    CAR_TELEMETRY_STRUCT,
    F125TelemetryAdapter,
    HEADER_STRUCT,
    LAP_DATA_STRUCT,
    PacketIds,
    TelemetryStateStore,
)


TEST_PACKET_FORMAT = 2025
TEST_GAME_YEAR = 25
TEST_MAJOR_VERSION = 1
TEST_MINOR_VERSION = 0
TEST_PACKET_VERSION = 1
TEST_SESSION_UID = 123456789
TEST_SESSION_TIME = 12.5
TEST_FRAME_IDENTIFIER = 10
TEST_OVERALL_FRAME_IDENTIFIER = 10
TEST_SECONDARY_PLAYER_INDEX = 255


def build_header(packet_id: int, player_index: int = 0) -> bytes:
    return HEADER_STRUCT.pack(
        TEST_PACKET_FORMAT,
        TEST_GAME_YEAR,
        TEST_MAJOR_VERSION,
        TEST_MINOR_VERSION,
        TEST_PACKET_VERSION,
        packet_id,
        TEST_SESSION_UID,
        TEST_SESSION_TIME,
        TEST_FRAME_IDENTIFIER,
        TEST_OVERALL_FRAME_IDENTIFIER,
        player_index,
        TEST_SECONDARY_PLAYER_INDEX,
    )


class TelemetryParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = TelemetryStateStore()
        self.adapter = F125TelemetryAdapter()

    def test_player_packets_update_snapshot(self) -> None:
        telemetry_entry = CAR_TELEMETRY_STRUCT.pack(
            312,
            0.95,
            0.01,
            0.0,
            0,
            8,
            12000,
            1,
            85,
            1023,
            630,
            628,
            640,
            638,
            98,
            99,
            100,
            101,
            92,
            93,
            94,
            95,
            108,
            21.5,
            21.6,
            22.1,
            22.2,
            0,
            0,
            0,
            0,
        )
        telemetry_packet = build_header(PacketIds.CAR_TELEMETRY) + telemetry_entry + (b"\x00" * CAR_TELEMETRY_STRUCT.size * 21)

        status_entry = CAR_STATUS_STRUCT.pack(
            0,
            0,
            2,
            56,
            0,
            28.4,
            110.0,
            18.6,
            15000,
            4000,
            8,
            1,
            300,
            16,
            16,
            10,
            -1,
            700000.0,
            120000.0,
            2800000.0,
            2,
            120000.0,
            25000.0,
            50000.0,
            0,
        )
        status_packet = build_header(PacketIds.CAR_STATUS) + status_entry + (b"\x00" * CAR_STATUS_STRUCT.size * 21)

        damage_entry = CAR_DAMAGE_STRUCT.pack(
            12.5,
            13.0,
            15.5,
            16.0,
            *([0] * 8),
            2,
            3,
            4,
            1,
            0,
            0,
            *([0] * 12),
        )
        damage_packet = build_header(PacketIds.CAR_DAMAGE) + damage_entry + (b"\x00" * CAR_DAMAGE_STRUCT.size * 21)

        for packet in (telemetry_packet, status_packet, damage_packet):
            self.adapter.parse_packet(packet, self.state)

        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["player"]["speed_kph"], 312)
        self.assertEqual(snapshot["player"]["gear"], 8)
        self.assertEqual(snapshot["player"]["fuel_in_tank_l"], 28.4)
        self.assertEqual(snapshot["player"]["ers_store_energy_j"], 2800000.0)
        self.assertEqual(snapshot["player"]["tyre_wear_pct"]["FL"], 15.5)
        self.assertEqual(snapshot["player"]["tyre_pressure_psi"]["FR"], 22.2)

    def test_lap_packet_derives_gaps_and_lap_gain(self) -> None:
        entries = []
        entries.append(
            LAP_DATA_STRUCT.pack(
                90500,
                30000,
                29000,
                0,
                30000,
                0,
                1500,
                4500,
                4200.0,
                54200.0,
                0.0,
                2,
                6,
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                1,
                2,
                0,
                0,
                0,
                0,
            )
        )
        entries.append(
            LAP_DATA_STRUCT.pack(
                90000,
                30100,
                28900,
                0,
                30000,
                0,
                0,
                3000,
                4250.0,
                54250.0,
                0.0,
                1,
                6,
                0,
                1,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                2,
                0,
                0,
                0,
                0,
            )
        )
        entries.extend([b"\x00" * LAP_DATA_STRUCT.size for _ in range(20)])
        first_lap_packet = build_header(PacketIds.LAP_DATA) + b"".join(entries)
        self.adapter.parse_packet(first_lap_packet, self.state)

        entries[0] = LAP_DATA_STRUCT.pack(
            89000,
            1000,
            29000,
            0,
            30000,
            0,
            1200,
            4200,
            200.0,
            60000.0,
            0.0,
            2,
            7,
            0,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            1,
            2,
            0,
            0,
            0,
            0,
        )
        second_lap_packet = build_header(PacketIds.LAP_DATA) + b"".join(entries)
        self.adapter.parse_packet(second_lap_packet, self.state)

        snapshot = self.state.snapshot()
        self.assertEqual(snapshot["player"]["position"], 2)
        self.assertAlmostEqual(snapshot["race"]["gap_to_car_ahead_s"], 1.2)
        self.assertAlmostEqual(snapshot["race"]["gap_to_leader_s"], 4.2)
        self.assertEqual(snapshot["race"]["car_ahead"]["position"], 1)
        self.assertAlmostEqual(snapshot["player"]["lap_gain_seconds"], 1.5)
        self.assertEqual(snapshot["session"]["lap_history_ms"]["6"], 89000)


if __name__ == "__main__":
    unittest.main()
