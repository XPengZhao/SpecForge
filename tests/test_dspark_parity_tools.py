"""CPU/stdlib checks for the diagnostic installer and case selection."""
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ParityToolsTest(unittest.TestCase):
    def test_greedy_acceptance_stops_at_first_rejection(self):
        tool = load("compare_dspark_fixed_prefix")
        self.assertEqual(tool.matching_prefix([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(tool.matching_prefix([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(tool.matching_prefix([9, 2, 3], [1, 2, 3]), 0)
        with self.assertRaises(ValueError):
            tool.matching_prefix([1], [1, 2])

    def test_anchors_do_not_cross_loss_mask_gaps(self):
        tool = load("compare_dspark_fixed_prefix")
        mask = [0] + [1] * 10 + [0] + [1] * 9
        anchors = tool.select_anchors(mask, 7, 3)
        self.assertEqual(anchors, [1, 3, 13])
        for anchor in anchors:
            self.assertTrue(all(mask[anchor:anchor + 8]))

    def test_install_idempotent_restore_and_protect_edits(self):
        source = "class DSparkSpeculator(Base):\n    def _sample_sequential(self, num_reqs, head_hidden):\n        return head_hidden\n"
        installer = load("install_dspark_parity_hook")
        tree = ast.parse(installer.instrument(source))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        self.assertEqual([node.name for node in cls.body], ["propose", "_sample_sequential"])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"
            target.parent.mkdir(parents=True)
            target.write_text(source)
            command = [sys.executable, str(ROOT / "scripts/install_dspark_parity_hook.py"), "--vllm-root", directory]
            subprocess.run(command, check=True, capture_output=True)
            installed = target.read_text()
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(target.read_text(), installed)
            target.write_text(installed + "# manual edit\n")
            result = subprocess.run(command + ["--restore"], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("manual edit", target.read_text())
            target.write_text(installed)
            subprocess.run(command + ["--restore"], check=True, capture_output=True)
            self.assertEqual(target.read_text(), source)
            self.assertFalse(target.with_name("parity_debug.py").exists())


if __name__ == "__main__":
    unittest.main()
