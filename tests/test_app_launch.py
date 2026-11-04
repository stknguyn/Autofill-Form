"""
tests/test_app_launch.py - Automated Launch & Compilation Verification for app.py.
Verifies:
- app.py compiles cleanly without syntax errors
- Streamlit AppTest initializes and renders cleanly without unhandled runtime exceptions
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

APP_PATH = REPO_ROOT / "app.py"


class TestAppLaunch(unittest.TestCase):
    """Automated launch and execution verification for the Streamlit dashboard."""

    def test_app_syntax_and_compilation(self) -> None:
        """Verify that app.py compiles cleanly without syntax or indentation errors."""
        self.assertTrue(APP_PATH.exists(), f"app.py not found at {APP_PATH}")
        with open(APP_PATH, "r", encoding="utf-8") as f:
            source = f.read()

        # 1. Parse AST
        tree = ast.parse(source, filename=str(APP_PATH))
        self.assertIsNotNone(tree, "Failed to parse app.py AST")

        # 2. Compile code object
        code_obj = compile(source, str(APP_PATH), "exec")
        self.assertIsNotNone(code_obj, "Failed to compile app.py bytecode")

    def test_app_clean_initialization_via_apptest(self) -> None:
        """Verify that app.py initializes cleanly in headless Streamlit AppTest without unhandled exceptions."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(APP_PATH), default_timeout=30)
        at.run()

        # Verify zero runtime exceptions during app execution
        self.assertEqual(
            len(at.exception),
            0,
            f"Unhandled exceptions during app launch: {[e.message for e in at.exception]}",
        )


if __name__ == "__main__":
    unittest.main()
