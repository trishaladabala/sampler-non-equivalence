"""E2E Test Suite for ELF Research Roadmap Notebooks.

Validates that all roadmap notebooks:
  1. elf_p0_harness.ipynb (Prompt 0 reference harness)
  2. elf_p1a_ecp_sweep.ipynb (Prompt 1A ECP gates & sweeps)
  3. elf_p1b_ecp_ft.ipynb (Prompt 1B ECP fine-tuning)
  4. elf_p2a_cmlc_train.ipynb (Prompt 2A CMLC retrofit & probe)
  5. elf_p2b_cmlc_eval.ipynb (Prompt 2B CMLC sampler & matrix)
  6. elf_p3_expanding_pilot.ipynb (Stretch Proposal P3 Pilot)
  7. elf_p4_frontier.ipynb (Prompt 4 Frontier analysis)

Verification Criteria:
  - File existence in project root
  - Strict compliance with nbformat v4 schema
  - Zero SyntaxError and zero SyntaxWarning across all code cells under ast.parse and compile()
  - Colab self-containment cells (Drive mount, zip extract/git fallback, pip install, sys.path, elfx/ writefile)
"""

import ast
import json
import os
import sys
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

ROADMAP_NOTEBOOKS = [
    "elf_p1a_ecp_sweep.ipynb",
    "elf_p1b_ecp_ft.ipynb",
    "elf_p2a_cmlc_train.ipynb",
    "elf_p2b_cmlc_eval.ipynb",
    "elf_p3_expanding_pilot.ipynb",
    "elf_p4_frontier.ipynb",
]

REFERENCE_NOTEBOOK = "elf_p0_harness.ipynb"


def clean_code_cell_source(source: str) -> str:
    """Strips IPython line magics and shell escapes while preserving Python code in %%writefile."""
    lines = source.splitlines()
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Cell magic: %%writefile <path> -> drop the magic header line, keep module code
        if stripped.startswith("%%writefile"):
            continue
        # Line magics and shell commands: %pip, %cd, !git, !pip, etc.
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        clean_lines.append(line)
    return "\n".join(clean_lines)


def validate_nbformat_v4_schema(nb_data: Dict[str, Any], path: Path) -> List[str]:
    """Validates structural conformity to nbformat v4 specification."""
    errors = []
    if nb_data.get("nbformat") != 4:
        errors.append(f"Expected nbformat == 4, got {nb_data.get('nbformat')}")
    if "cells" not in nb_data or not isinstance(nb_data["cells"], list):
        errors.append("Missing or invalid 'cells' list in notebook JSON")
        return errors

    if "metadata" not in nb_data or not isinstance(nb_data["metadata"], dict):
        errors.append("Missing or invalid top-level 'metadata' dictionary")

    for i, cell in enumerate(nb_data["cells"]):
        if not isinstance(cell, dict):
            errors.append(f"Cell {i} is not a dictionary")
            continue
        cell_type = cell.get("cell_type")
        if cell_type not in ("code", "markdown", "raw"):
            errors.append(f"Cell {i} has invalid cell_type: {cell_type}")
        if "source" not in cell:
            errors.append(f"Cell {i} missing 'source' attribute")
        if cell_type == "code":
            if "outputs" not in cell or not isinstance(cell["outputs"], list):
                errors.append(f"Code cell {i} missing 'outputs' list")
            if "execution_count" not in cell:
                errors.append(f"Code cell {i} missing 'execution_count'")

    return errors


def validate_ast_and_compilation(nb_data: Dict[str, Any], path: Path) -> List[str]:
    """Parses and compiles all code cells with ast.parse and compile() under strict warning filters."""
    errors = []
    for i, cell in enumerate(nb_data.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            raw_source = "".join(raw_source)

        cleaned_source = clean_code_cell_source(raw_source)
        if not cleaned_source.strip():
            continue

        filename = f"{path.name}#cell_{i}.py"
        try:
            ast.parse(cleaned_source, filename=filename)
        except SyntaxError as se:
            errors.append(f"Cell {i} ast.parse SyntaxError: {se} (line {se.lineno})")
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                compile(cleaned_source, filename=filename, mode="exec")
        except SyntaxWarning as sw:
            errors.append(f"Cell {i} compile SyntaxWarning under strict filter: {sw}")
        except Exception as ex:
            errors.append(f"Cell {i} compilation error: {ex}")

    return errors


def validate_colab_self_containment(nb_data: Dict[str, Any], path: Path) -> Dict[str, bool]:
    """Verifies presence of Colab bootstrap components for fresh VM execution."""
    all_text = "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in nb_data.get("cells", [])
    )

    checks = {
        "drive_mount": ("drive.mount" in all_text) or ("google.colab" in all_text),
        "extract_or_clone": (
            ("elf_code.zip" in all_text)
            or ("git clone" in all_text)
            or ("ELF_GIT_FALLBACK" in all_text)
            or ("ELF_SRC" in all_text)
        ),
        "pip_install": (
            ("pip install" in all_text)
            or ("'pip', 'install'" in all_text)
            or ('"pip", "install"' in all_text)
        ),
        "sys_path": "sys.path" in all_text,
        "writefile_elfx": ("%%writefile" in all_text) and ("elfx" in all_text),
    }
    return checks


class TestAllRoadmaps(unittest.TestCase):
    """Test suite validating all Colab research roadmap notebooks."""

    def _check_notebook(self, nb_filename: str):
        path = WORKSPACE_ROOT / nb_filename
        self.assertTrue(path.exists(), f"Notebook file missing: {path}")

        # 1. Parse JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
        except Exception as e:
            self.fail(f"Failed to load JSON from {nb_filename}: {e}")

        # 2. Schema check
        schema_errors = validate_nbformat_v4_schema(nb_data, path)
        self.assertEqual(
            len(schema_errors),
            0,
            f"Schema validation failed for {nb_filename}:\n" + "\n".join(schema_errors),
        )

        # 3. AST and compile check
        ast_errors = validate_ast_and_compilation(nb_data, path)
        self.assertEqual(
            len(ast_errors),
            0,
            f"AST / compilation validation failed for {nb_filename}:\n" + "\n".join(ast_errors),
        )

        # 4. Colab self-containment check
        containment = validate_colab_self_containment(nb_data, path)
        failed_containment = [k for k, v in containment.items() if not v]
        self.assertEqual(
            len(failed_containment),
            0,
            f"Colab self-containment checks failed for {nb_filename}: {failed_containment}",
        )

    def test_p0_reference_harness(self):
        """Prompt 0 reference harness must be valid and conformant."""
        self._check_notebook(REFERENCE_NOTEBOOK)

    def test_p1a_ecp_sweep(self):
        """Prompt 1A notebook: elf_p1a_ecp_sweep.ipynb."""
        self._check_notebook("elf_p1a_ecp_sweep.ipynb")

    def test_p1b_ecp_ft(self):
        """Prompt 1B notebook: elf_p1b_ecp_ft.ipynb."""
        self._check_notebook("elf_p1b_ecp_ft.ipynb")

    def test_p2a_cmlc_train(self):
        """Prompt 2A notebook: elf_p2a_cmlc_train.ipynb."""
        self._check_notebook("elf_p2a_cmlc_train.ipynb")

    def test_p2b_cmlc_eval(self):
        """Prompt 2B notebook: elf_p2b_cmlc_eval.ipynb."""
        self._check_notebook("elf_p2b_cmlc_eval.ipynb")

    def test_p3_expanding_pilot(self):
        """Stretch Proposal P3 Pilot notebook: elf_p3_expanding_pilot.ipynb."""
        self._check_notebook("elf_p3_expanding_pilot.ipynb")

    def test_p4_frontier(self):
        """Prompt 4 notebook: elf_p4_frontier.ipynb."""
        self._check_notebook("elf_p4_frontier.ipynb")

    def test_all_six_roadmaps_inventory(self):
        """Verifies that all 6 roadmap notebooks exist in project root."""
        missing = [nb for nb in ROADMAP_NOTEBOOKS if not (WORKSPACE_ROOT / nb).exists()]
        self.assertEqual(
            len(missing),
            0,
            f"Missing required roadmap notebooks: {missing}",
        )


def run_summary():
    """Runs validation report across all notebooks and outputs formatted table."""
    print("=" * 78)
    print("ELF ROADMAP NOTEBOOK VALIDATION AUDIT")
    print("=" * 78)

    all_nbs = [REFERENCE_NOTEBOOK] + ROADMAP_NOTEBOOKS
    results = []

    for nb_name in all_nbs:
        path = WORKSPACE_ROOT / nb_name
        if not path.exists():
            results.append((nb_name, "MISSING", 0, 0, "File does not exist"))
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                nb_data = json.load(f)
        except Exception as e:
            results.append((nb_name, "INVALID_JSON", 0, 0, str(e)))
            continue

        schema_errs = validate_nbformat_v4_schema(nb_data, path)
        ast_errs = validate_ast_and_compilation(nb_data, path)
        containment = validate_colab_self_containment(nb_data, path)

        code_cells = sum(1 for c in nb_data.get("cells", []) if c.get("cell_type") == "code")
        total_cells = len(nb_data.get("cells", []))

        status = "PASS"
        issues = []
        if schema_errs:
            status = "SCHEMA_ERR"
            issues.extend(schema_errs)
        if ast_errs:
            status = "SYNTAX_ERR"
            issues.extend(ast_errs)
        missing_containment = [k for k, v in containment.items() if not v]
        if missing_containment:
            status = "CONTAINMENT_ERR"
            issues.append(f"Missing setup: {missing_containment}")

        detail = "OK" if status == "PASS" else "; ".join(issues)[:45]
        results.append((nb_name, status, total_cells, code_cells, detail))

    print(f"{'Notebook':<32} {'Status':<16} {'Cells':<6} {'Code':<6} {'Details'}")
    print("-" * 78)
    for nb_name, status, total_cells, code_cells, detail in results:
        print(f"{nb_name:<32} {status:<16} {total_cells:<6} {code_cells:<6} {detail}")
    print("=" * 78)


if __name__ == "__main__":
    if "--summary" in sys.argv:
        run_summary()
    else:
        run_summary()
        unittest.main()
