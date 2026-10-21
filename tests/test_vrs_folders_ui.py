"""UI regression checks use only Node's standard library when available."""
from pathlib import Path
import shutil
import subprocess
import unittest


class FolderUITests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "Node.js is optional for offline UI tests")
    def test_user_flows(self):
        script = Path(__file__).with_name("vrs_folders_ui.mjs")
        result = subprocess.run([shutil.which("node"), str(script)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
