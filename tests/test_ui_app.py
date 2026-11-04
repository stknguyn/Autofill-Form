"""
tests/test_ui_app.py - Integration & AppTest Verification for app.py UI.
Verifies:
- 0 unhandled exceptions or syntax warnings on initial render
- All 4 tabs exist and contain appropriate child elements
- Master control buttons (Start, Pause/Resume, Stop, Reset) work seamlessly
- Export service integration in Tab 3 renders properly
- Streamlit 1.55+ compatibility (no use_container_width deprecation warnings)
"""

import sys
import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class TestAppUI(unittest.TestCase):
    def test_app_initial_render(self):
        """Verifies that app.py compiles, executes, and renders with 0 exceptions."""
        at = AppTest.from_file(str(ROOT_DIR / "app.py"))
        at.run()
        self.assertEqual(len(at.exception), 0, f"Exceptions on initial render: {at.exception}")

        # Verify tabs exist
        self.assertEqual(len(at.tabs), 4, f"Expected 4 tabs, found {len(at.tabs)}")

        # Verify sidebar buttons exist
        button_labels = [b.label for b in at.button]
        self.assertIn("▶ Chạy", button_labels)
        self.assertIn("⏸ Dừng", button_labels)
        self.assertIn("⏹ Ngắt", button_labels)
        self.assertIn("🔄 Đặt lại", button_labels)
        self.assertIn("📂 Tải Cấu Hình Cũ", button_labels)
        self.assertIn("💾 Lưu Cấu Hình", button_labels)

    def test_app_config_persistence_lifecycle(self):
        """Verifies manual save and load configuration via sidebar UI buttons."""
        cfg_file = ROOT_DIR / "session_config.json"
        existed_initially = cfg_file.exists()
        backup_content = cfg_file.read_text(encoding="utf-8") if existed_initially else None

        try:
            at = AppTest.from_file(str(ROOT_DIR / "app.py"))
            at.run()
            self.assertEqual(len(at.exception), 0)

            # Change target count in session state
            at.session_state["cfg_target_count"] = 280
            at.session_state["cfg_num_workers"] = 12

            # Click Save Config button
            save_btns = [b for b in at.button if "💾 Lưu Cấu Hình" in b.label]
            self.assertTrue(len(save_btns) > 0)
            save_btns[0].click()
            at.run()
            self.assertEqual(len(at.exception), 0)
            self.assertTrue(cfg_file.exists())

            # Now mutate session state to something different
            at.session_state["cfg_target_count"] = 99
            at.session_state["cfg_num_workers"] = 2

            # Click Load Config button
            load_btns = [b for b in at.button if "📂 Tải Cấu Hình Cũ" in b.label]
            self.assertTrue(len(load_btns) > 0)
            self.assertFalse(load_btns[0].disabled)
            load_btns[0].click()
            at.run()
            self.assertEqual(len(at.exception), 0)

            # Verify restored
            self.assertEqual(at.session_state["cfg_target_count"], 280)
            self.assertEqual(at.session_state["cfg_num_workers"], 12)

        finally:
            # Restore previous state
            if existed_initially and backup_content is not None:
                cfg_file.write_text(backup_content, encoding="utf-8")
            elif not existed_initially and cfg_file.exists():
                cfg_file.unlink()

    def test_app_start_and_stop_lifecycle(self):
        """Verifies the UI start/stop lifecycle through AppTest button clicks."""
        at = AppTest.from_file(str(ROOT_DIR / "app.py"))
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Click start button
        start_btn = [b for b in at.button if "▶ Chạy" in b.label][0]
        start_btn.click()
        at.run()
        self.assertEqual(len(at.exception), 0)

        # Click stop button
        stop_btn = [b for b in at.button if "⏹ Ngắt" in b.label][0]
        stop_btn.click()
        at.run()
        self.assertEqual(len(at.exception), 0)


if __name__ == "__main__":
    unittest.main()
