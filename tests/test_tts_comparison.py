import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import importlib.util

spec = importlib.util.spec_from_file_location('compare_tts', Path(__file__).resolve().parents[1] / 'debug/compare_tts_endings.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def test_pairs_change_only_intended_factors(self):
        cases = comparison.make_cases('Маши́на едет.', [100001], 1.0, 0.5)
        pairs = {c['variant']: c['payload'] for c in cases}
        self.assertEqual(len(cases), 5)
        baseline = pairs['stress_baseline']
        self.assertEqual(baseline, pairs['stress_repeat'])
        self.assertEqual({**baseline, 'text': 'Машина едет.'}, pairs['plain_baseline'])
        self.assertEqual({**baseline, 'exaggeration': 0.5}, pairs['stress_lower'])
        self.assertFalse(baseline['split_text'])
        self.assertEqual(baseline['seed'], 100001)

    def test_wav_bytes_preserved_and_no_forced_eos_inferred(self):
        cases = comparison.make_cases('Маши́на.', [7], 1.0, 0.5)[:1]
        content = b'RIFF\x00\x00\x00\x00WAVEraw-audio-tail'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            with patch.object(comparison.dp.requests, 'post', return_value=Mock(ok=True, content=content)), \
                 patch.object(comparison, 'audio_info', return_value={'duration_seconds': 5.4}):
                records = comparison.execute(cases, output)
            self.assertEqual((output / records[0]['file']).read_bytes(), content)
            self.assertNotIn('forced_eos_from_server_log', records[0])
            self.assertEqual(records[0]['status'], 'completed')

    def test_stop_after_timeout(self):
        cases = comparison.make_cases('Маши́на.', [7], 1.0, 0.5)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(comparison.dp.requests, 'post', side_effect=TimeoutError) as request:
                records = comparison.execute(cases, Path(folder))
            self.assertEqual(len(records), 1)
            self.assertEqual(request.call_count, 1)
            self.assertEqual(records[0]['status'], 'failed')

    def test_production_request_defaults_unchanged(self):
        payload = comparison.dp.build_chatterbox_payload('text')
        self.assertEqual(payload['seed'], 0)
        self.assertTrue(payload['split_text'])
        self.assertEqual(payload['text'], 'text')

    def test_unmarked_input_rejected(self):
        with self.assertRaises(ValueError):
            comparison.make_cases('Машина.', [1], 1.0, 0.5)
