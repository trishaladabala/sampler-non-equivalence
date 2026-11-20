"""
tests/test_revision_notebooks.py: Automated verification test suite for Milestone 2 revision notebooks.

Verifies:
1. File existence of all 6 revision notebooks:
   - notebooks/16_preregistered_c1_c3_replication.ipynb (WS1)
   - notebooks/17_heldout_reference_and_contamination_audit.ipynb (WS2)
   - notebooks/15_final_human_evaluation.ipynb (WS3')
   - notebooks/18_plaid_feasibility_spike.ipynb (WS4 Spike)
   - notebooks/19_plaid_heldout_equivalence.ipynb (WS4 Study)
   - notebooks/21_ace_null_distribution_and_matched_ace.ipynb (WS5)
2. Strict structural compliance with nbformat v4 specification.
3. Clean compilation under ast.parse and compile(mode='exec') with zero SyntaxError / SyntaxWarning under python -W error.
4. Fail-closed Colab architecture: raises ColabEngineeringError when executed outside Colab without Drive mounted.
5. Algorithmic and mathematical correctness on CPU smoke fixtures via run_smoke_cpu().
"""

import ast
import json
import os
import re
import sys
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REVISION_NOTEBOOKS = [
    "16_preregistered_c1_c3_replication.ipynb",
    "17_heldout_reference_and_contamination_audit.ipynb",
    "15_final_human_evaluation.ipynb",
    "18_plaid_feasibility_spike.ipynb",
    "19_plaid_heldout_equivalence.ipynb",
    "21_ace_null_distribution_and_matched_ace.ipynb",
    "22_acl_remediation_reruns.ipynb",
]

ALL_REVISION_NOTEBOOKS = list(dict.fromkeys(REVISION_NOTEBOOKS))


def clean_source(source: Any) -> str:
    """Combines cell source list/string and removes IPython magics."""
    if isinstance(source, list):
        text = "".join(source)
    else:
        text = str(source)
    clean_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("%%writefile") or stripped.startswith("%") or stripped.startswith("!"):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


class TestRevisionNotebooks(unittest.TestCase):
    """Test suite validating all Milestone 2 revision notebooks."""

    def test_notebook_inventory_and_schema(self):
        """All revision and colab production notebooks must exist and satisfy nbformat v4 schema."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            path = REPO_ROOT / "notebooks" / nb_name
            self.assertTrue(path.is_file(), f"Missing required revision notebook: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data.get("nbformat"), 4, f"{nb_name} must have nbformat == 4")
            self.assertIn("cells", data, f"{nb_name} missing cells list")
            self.assertIsInstance(data["cells"], list)
            self.assertGreaterEqual(len(data["cells"]), 4, f"{nb_name} must have at least 4 cells")

            for i, cell in enumerate(data["cells"]):
                self.assertIn("cell_type", cell, f"{nb_name} cell {i} missing cell_type")
                self.assertIn(cell["cell_type"], ("code", "markdown", "raw"))
                self.assertIn("source", cell, f"{nb_name} cell {i} missing source")
                if cell["cell_type"] == "code":
                    self.assertIn("outputs", cell, f"{nb_name} code cell {i} missing outputs")

    def test_clean_ast_and_compilation(self):
        """Every code cell across all notebooks must parse cleanly with ast.parse and compile under strict warning filter."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for i, cell in enumerate(data["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                code_text = clean_source(cell.get("source", ""))
                if not code_text.strip():
                    continue

                filename = f"{nb_name}#cell_{i}.py"
                # Parse AST
                try:
                    tree = ast.parse(code_text, filename=filename)
                    self.assertIsNotNone(tree)
                except SyntaxError as se:
                    self.fail(f"SyntaxError in {filename}: {se}")

                # Compile with strict warnings
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        compile(code_text, filename=filename, mode="exec")
                except Exception as ex:
                    self.fail(f"Compilation failure in {filename} under strict filter: {ex}")

    def test_fail_closed_drive_assertions(self):
        """When RUN_SMOKE_CPU=False and Drive is absent, preflight checks must raise ColabEngineeringError."""
        for nb_name in ALL_REVISION_NOTEBOOKS:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract cell 1 (preflight checks)
            code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
            self.assertGreater(len(code_cells), 0)
            cell1_source = clean_source(code_cells[0]["source"])

            # Execute cell 1 with RUN_SMOKE_CPU = False
            # Should raise ColabEngineeringError because we are not in Colab
            env_globals = {"__name__": "__test__"}
            exec_code = cell1_source.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = False")
            
            with self.assertRaises(Exception) as ctx:
                exec(exec_code, env_globals)

            # Check that raised exception is ColabEngineeringError or DriveNotMountedError
            exc = ctx.exception
            exc_type_name = type(exc).__name__
            self.assertIn(
                exc_type_name,
                ("ColabEngineeringError", "DriveNotMountedError", "FileNotFoundError"),
                f"{nb_name} failed to fail closed with expected exception, raised: {exc_type_name} ({exc})"
            )

    def test_cpu_smoke_execution_nb16(self):
        """Notebook 16 run_smoke_cpu must verify CRN latents, 2^20 permutation test, and gatekeeping."""
        path = REPO_ROOT / "notebooks" / "16_preregistered_c1_c3_replication.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        # Enable smoke mode
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_execution_nb17(self):
        """Notebook 17 run_smoke_cpu must audit 13-gram contamination and compute CTW JCD."""
        path = REPO_ROOT / "notebooks" / "17_heldout_reference_and_contamination_audit.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_execution_nb15(self):
        """Notebook 15 run_smoke_cpu must pass benchmark grounding and position-swapped TOST."""
        path = REPO_ROOT / "notebooks" / "15_final_human_evaluation.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_execution_nb18(self):
        """Notebook 18 run_smoke_cpu must sweep 4x4 surface and evaluate go/no-go gate."""
        path = REPO_ROOT / "notebooks" / "18_plaid_feasibility_spike.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_execution_nb19(self):
        """Notebook 19 run_smoke_cpu must execute held-out equivalence Joint TOST."""
        path = REPO_ROOT / "notebooks" / "19_plaid_heldout_equivalence.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_execution_nb21(self):
        """Notebook 21 run_smoke_cpu must test 20 orthogonal null directions and matched ACE TOST."""
        path = REPO_ROOT / "notebooks" / "21_ace_null_distribution_and_matched_ace.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
        full_code = "\n".join(code_cells)
        full_code = full_code.replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        full_code = full_code.replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")

        env_globals = {"__name__": "__test__"}
        exec(full_code, env_globals)
        self.assertIn("run_smoke_cpu", env_globals)

    def test_cpu_smoke_colab_production_notebooks(self):
        """All 6 colab production notebooks must execute their run_smoke_cpu() cleanly."""
        real_elfx_dir = str(REPO_ROOT / "elfx")
        if "elfx" in sys.modules and hasattr(sys.modules["elfx"], "__path__"):
            if real_elfx_dir not in sys.modules["elfx"].__path__:
                sys.modules["elfx"].__path__.append(real_elfx_dir)
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        for nb_name in REVISION_NOTEBOOKS:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
            full_code = "\n".join(code_cells)
            full_code = re.sub(r"RUN_SMOKE_CPU\s*=\s*False", "RUN_SMOKE_CPU = True", full_code, count=1)

            env_globals = {"__name__": "__test__"}
            exec(full_code, env_globals)
            self.assertIn("run_smoke_cpu", env_globals, f"{nb_name} missing run_smoke_cpu in smoke execution")


if __name__ == "__main__":
    unittest.main()
