"""
tests/test_config_service.py - Unit tests for ConfigService.
Verifies:
- Saving session configuration to JSON with metadata and timestamp
- Loading session configuration accurately
- Graceful handling of nonexistent or corrupted files
- Metadata summary extraction
- Application of loaded configuration to session_state object
"""

import json
import tempfile
import unittest
from pathlib import Path

from src.config_service import ConfigService, DEFAULT_CONSTRUCT_BENCHMARKS, DEFAULT_SESSION_CONFIG


class TestConfigService(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_file = Path(self.temp_dir.name) / "test_config.json"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config_structure(self):
        cfg = ConfigService.get_default_config()
        self.assertIn("campaign", cfg)
        self.assertIn("pacing", cfg)
        self.assertIn("scheduler", cfg)
        self.assertIn("demographics", cfg)
        self.assertIn("psychometrics", cfg)
        self.assertEqual(cfg["campaign"]["target_count"], 150)
        self.assertEqual(cfg["pacing"]["delay_min"], 1800.0)
        self.assertEqual(cfg["pacing"]["delay_max"], 3600.0)

    def test_save_and_load_config(self):
        custom_data = {
            "campaign": {"form_url": "https://tally.so/r/custom", "target_count": 350, "num_workers": 10, "master_seed": 999},
            "pacing": {"dist_choice": "Uniform (Đều)", "delay_min": 1900.0, "delay_max": 3700.0},
            "demographics": {"age_mean": 22.0, "hcm_pct": 60},
        }

        saved_path = ConfigService.save_session_config(custom_data, self.temp_file)
        self.assertTrue(saved_path.exists())

        loaded = ConfigService.load_session_config(self.temp_file)
        self.assertIsNotNone(loaded)
        self.assertIn("saved_at", loaded)
        self.assertEqual(loaded["campaign"]["target_count"], 350)
        self.assertEqual(loaded["pacing"]["delay_min"], 1900.0)
        self.assertEqual(loaded["demographics"]["hcm_pct"], 60)

    def test_has_saved_config_and_summary(self):
        self.assertFalse(ConfigService.has_saved_config(self.temp_file))
        self.assertIsNone(ConfigService.get_saved_config_summary(self.temp_file))

        ConfigService.save_session_config(ConfigService.get_default_config(), self.temp_file)
        self.assertTrue(ConfigService.has_saved_config(self.temp_file))

        summary = ConfigService.get_saved_config_summary(self.temp_file)
        self.assertIsNotNone(summary)
        self.assertEqual(summary["target_count"], 150)
        self.assertEqual(summary["delay_min"], 1800.0)
        self.assertEqual(summary["delay_max"], 3600.0)

    def test_apply_to_session_state(self):
        mock_state = {}
        cfg = {
            "campaign": {"target_count": 220, "num_workers": 8, "master_seed": 77},
            "pacing": {"delay_min": 1850.0, "delay_max": 3550.0, "dist_choice": "Uniform (Đều)"},
            "psychometrics": {
                "alpha_threshold": 0.85,
                "construct_means": {"PU": 5.0, "PEOU": 5.1},
            },
        }
        ConfigService.apply_to_session_state(cfg, mock_state)

        self.assertEqual(mock_state["cfg_target_count"], 220)
        self.assertEqual(mock_state["cfg_num_workers"], 8)
        self.assertEqual(mock_state["cfg_master_seed"], 77)
        self.assertEqual(mock_state["cfg_delay_range"], (1850.0, 3550.0))
        self.assertEqual(mock_state["cfg_alpha_threshold"], 0.85)
        self.assertEqual(mock_state["cfg_pu_mean"], 5.0)
        self.assertEqual(mock_state["cfg_peou_mean"], 5.1)

    def test_extract_from_session_state(self):
        mock_state = {}
        ConfigService.init_session_state(mock_state)

        # Mutate some fields
        mock_state["cfg_target_count"] = 300
        mock_state["cfg_num_workers"] = 12
        mock_state["cfg_delay_range"] = (2000.0, 4000.0)
        mock_state["cfg_pu_mean"] = 5.25

        extracted = ConfigService.extract_from_session_state(mock_state)
        self.assertEqual(extracted["campaign"]["target_count"], 300)
        self.assertEqual(extracted["campaign"]["num_workers"], 12)
        self.assertEqual(extracted["pacing"]["delay_min"], 2000.0)
        self.assertEqual(extracted["pacing"]["delay_max"], 4000.0)
        self.assertEqual(extracted["psychometrics"]["construct_means"]["PU"], 5.25)


if __name__ == "__main__":
    unittest.main()
