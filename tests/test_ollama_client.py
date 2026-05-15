import json
import unittest
from unittest.mock import patch

from ai_race_engineer.ollama_client import OllamaRaceEngineer, SYSTEM_PROMPT


class OllamaClientTests(unittest.TestCase):
    def test_build_messages_embed_driver_question_and_telemetry(self) -> None:
        engineer = OllamaRaceEngineer(model="phi3")
        telemetry = {"player": {"fuel_in_tank_l": 12.3, "ers_store_energy_j": 2500000}}
        messages = engineer.build_messages("Wie sehen die Reifen aus?", telemetry)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("maximal in 1 bis 2 kurzen Sätzen", SYSTEM_PROMPT)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["driver_question"], "Wie sehen die Reifen aus?")
        self.assertEqual(payload["telemetry"], telemetry)
        self.assertEqual(payload["response_style"]["language"], "de")

    @patch("urllib.request.urlopen")
    def test_answer_reads_chat_completion_response(self, mocked_urlopen) -> None:
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "Box this lap."}}]}).encode("utf-8")

        mocked_urlopen.return_value = FakeResponse()
        engineer = OllamaRaceEngineer(model="phi3")
        reply = engineer.answer("Boxenstopp?", {"player": {"current_lap": 10}})
        self.assertEqual(reply, "Box this lap.")


if __name__ == "__main__":
    unittest.main()
