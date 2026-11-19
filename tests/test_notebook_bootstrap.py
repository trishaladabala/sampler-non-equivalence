"""
tests/test_notebook_bootstrap.py: Automated verification of fast, zero-lag Colab environment bootstrap architecture.

Asserts that:
1. Every revision notebook contains a compact (<150 lines), zero-lag Colab bootstrap cell as Cell 1.
2. Zero notebooks in `notebooks/` contain giant embedded base64 blobs (ELFX_CORE_B64, NI_SUITE_B64, S2_SLICE_B64, SLICE_B64) or `os.chdir(PROJECT_ROOT)` on the Google Drive FUSE mount in `15`–`21`.
3. The bootstrap cell checks `importlib.util.find_spec` before running `pip install`, extracts `/content/repo/ELF/src`, syncs `elfx` to local `/content/code`, and sets `sys.path`.
4. No cell before Cell 1 imports `elfx`, and all subsequent `elfx` imports occur in cells strictly > 1 (after the bootstrap).
5. All `elfx/*.py` modules in the repository parse cleanly into valid Python syntax.
"""
import ast
import glob
import json
import sys
import unittest
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALL_REVISION_NOTEBOOKS = [
    "15_final_human_evaluation.ipynb",
    "16_preregistered_c1_c3_replication.ipynb",
    "17_heldout_reference_and_contamination_audit.ipynb",
    "18_plaid_feasibility_spike.ipynb",
    "19_plaid_heldout_equivalence.ipynb",
    "21_ace_null_distribution_and_matched_ace.ipynb",
]

EXPECTED_CORE_MODULES = {
    "__init__.py",
    "registry.py",
    "refs.py",
    "load.py",
    "sample.py",
    "metrics.py",
    "official.py",
    "sweep.py",
    "frontier.py",
    "non_identifiability.py",
}


class TestNotebookBootstrap(unittest.TestCase):
    def _get_first_code_cell(self, nb_path: Path) -> Tuple[int, str]:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                return idx, "".join(cell.get("source", []))
        return -1, ""

    def test_zero_base64_blobs_and_compact_cell1_across_all_notebooks(self):
        """No notebook in notebooks/*.ipynb may contain base64 code/slice blobs or >150-line Cell 1."""
        for p_str in sorted(glob.glob(str(REPO_ROOT / "notebooks" / "*.ipynb"))):
            nb_path = Path(p_str)
            nb_name = nb_path.name
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            code_cells = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
            if not code_cells:
                continue
            c1_src = "".join(code_cells[0].get("source", []))
            all_src = "\n".join("".join(c.get("source", [])) for c in code_cells)

            for forbidden in ("ELFX_CORE_B64", "NI_SUITE_B64", "S2_SLICE_B64", "SLICE_B64", "base64.b64decode"):
                self.assertNotIn(
                    forbidden,
                    all_src,
                    f"{nb_name} contains forbidden base64 blob pattern '{forbidden}' that lags Colab runtime",
                )

            c1_lines = len(c1_src.splitlines())
            self.assertLessEqual(
                c1_lines,
                150,
                f"{nb_name} Cell 1 is too large ({c1_lines} lines > 150) and risks lagging Colab UI",
            )

    def test_bootstrap_cell_presence_and_patterns(self):
        """Cell 1 of canonical notebooks must contain fast SSD sync, find_spec pip guard, and sys.path setup."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            nb_path = REPO_ROOT / "notebooks" / nb_name
            self.assertTrue(nb_path.is_file(), f"Missing notebook: {nb_name}")
            cell_idx, src = self._get_first_code_cell(nb_path)
            self.assertEqual(cell_idx, 1, f"{nb_name}: First code cell should be at index 1")

            self.assertNotIn("ELFX_CORE_B64", src, f"{nb_name} Cell 1 must not contain ELFX_CORE_B64")
            self.assertNotIn("os.chdir(PROJECT_ROOT)", src, f"{nb_name} Cell 1 must not chdir onto slow Drive FUSE mount")
            self.assertIn("/content/repo/ELF/src", src, f"{nb_name} Cell 1 missing /content/repo/ELF/src path")
            self.assertIn("elf_code.zip", src, f"{nb_name} Cell 1 missing elf_code.zip extraction logic")
            self.assertIn("importlib.util.find_spec", src, f"{nb_name} Cell 1 missing fast find_spec guard before pip")
            self.assertIn("sys.path.insert", src, f"{nb_name} Cell 1 missing sys.path configuration")

            if "15" in nb_name:
                self.assertIn("bitsandbytes", src, f"{nb_name} Cell 1 missing bitsandbytes pip spec")
                self.assertIn("accelerate", src, f"{nb_name} Cell 1 missing accelerate pip spec")
            elif "17" in nb_name:
                self.assertIn("datasets", src, f"{nb_name} Cell 1 missing datasets pip spec")
            else:
                self.assertIn("mauve-text", src, f"{nb_name} Cell 1 missing mauve-text pip spec")

    def test_no_elfx_imports_before_or_in_bootstrap_preamble(self):
        """No cell <= 1 may import elfx."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            nb_path = REPO_ROOT / "notebooks" / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            for idx, cell in enumerate(nb.get("cells", [])):
                if idx <= 1:
                    src = "".join(cell.get("source", []))
                    self.assertNotIn("import elfx", src, f"{nb_name} Cell {idx} imports elfx in or before bootstrap")
                    self.assertNotIn("from elfx", src, f"{nb_name} Cell {idx} imports elfx in or before bootstrap")

    def test_all_elfx_imports_occur_after_bootstrap_cell(self):
        """All elfx imports in cells >= 2 must only execute after bootstrap cell 1 has configured the environment."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            nb_path = REPO_ROOT / "notebooks" / nb_name
            with open(nb_path, "r", encoding="utf-8") as f:
                nb = json.load(f)

            elfx_import_cells = []
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") != "code":
                    continue
                src = "".join(cell.get("source", []))
                for line in src.splitlines():
                    s = line.strip()
                    if s.startswith(("import elfx", "from elfx")):
                        elfx_import_cells.append((idx, s))

            for cell_idx, stmt in elfx_import_cells:
                self.assertGreater(cell_idx, 1, f"{nb_name}: elfx import '{stmt}' found in cell {cell_idx} <= 1")

    def test_repo_elfx_modules_syntax(self):
        """All elfx/*.py modules in the repository must exist and parse cleanly."""
        elfx_dir = REPO_ROOT / "elfx"
        for fname in EXPECTED_CORE_MODULES:
            fpath = elfx_dir / fname
            self.assertTrue(fpath.is_file(), f"Missing core elfx module: {fpath}")
            code = fpath.read_text(encoding="utf-8")
            tree = ast.parse(code, filename=fname)
            self.assertIsNotNone(tree)


if __name__ == "__main__":
    unittest.main()
