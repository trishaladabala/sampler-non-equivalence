"""
tests/test_notebook_api_imports.py: Static AST validation of notebook API imports and call signatures.

Asserts that:
1. Every elfx submodule accessed via elfx.<submod>.<func> is explicitly imported in that notebook.
2. Every called function exists in the corresponding elfx submodule.
3. Every call satisfies the positional and keyword parameter signature of the target function.
"""
import ast
import json
import os
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REVISION_NOTEBOOKS = [
    "15_final_human_evaluation.ipynb",
    "16_preregistered_c1_c3_replication.ipynb",
    "17_heldout_reference_and_contamination_audit.ipynb",
    "18_plaid_feasibility_spike.ipynb",
    "19_plaid_heldout_equivalence.ipynb",
    "21_ace_null_distribution_and_matched_ace.ipynb",
]


def extract_elfx_signatures() -> Dict[Tuple[str, str], Tuple[Optional[List[str]], bool, bool, int]]:
    """
    Extracts signatures of all functions and classes across elfx/*.py.
    Returns: (module_name, func_name) -> (params, has_varkw, has_varargs, n_required)
    """
    sigs = {}
    elfx_dir = REPO_ROOT / "elfx"
    for py_path in elfx_dir.glob("*.py"):
        mod = py_path.stem
        try:
            tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        except Exception:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if isinstance(node, ast.ClassDef):
                    init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
                    if init is None:
                        sigs[(mod, node.name)] = (None, True, True, 0)
                        continue
                    a = init.args
                    params = [x.arg for x in a.args][1:] + [x.arg for x in a.kwonlyargs]
                    nreq = len(a.args) - 1 - len(a.defaults)
                else:
                    a = node.args
                    params = [x.arg for x in a.args] + [x.arg for x in a.kwonlyargs]
                    nreq = len(a.args) - len(a.defaults)
                sigs[(mod, node.name)] = (params, a.kwarg is not None, a.vararg is not None, nreq)
    return sigs


def check_notebook_api_calls(nb_path: Path, sigs: Dict[Tuple[str, str], Any]) -> List[str]:
    """Inspects a notebook's code cells for unimported submodules or invalid API calls."""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_lines = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for line in "".join(cell.get("source", [])).splitlines():
            stripped = line.strip()
            if not stripped.startswith(("%", "!")):
                code_lines.append(line)

    full_src = "\n".join(code_lines)
    try:
        tree = ast.parse(full_src, filename=str(nb_path))
    except SyntaxError as e:
        return [f"SYNTAX ERROR: {e}"]

    imported_mods: Set[str] = set()
    name_map: Dict[str, Tuple[str, ...]] = {}
    issues: List[str] = []

    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("elfx."):
                    submod = a.name.split(".")[1]
                    imported_mods.add(submod)
                    if a.asname:
                        name_map[a.asname] = ("MOD", submod)
        elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("elfx"):
            parts = n.module.split(".")
            if len(parts) == 1:
                for a in n.names:
                    imported_mods.add(a.name)
                    name_map[a.asname or a.name] = ("MOD", a.name)
            else:
                submod = parts[1]
                imported_mods.add(submod)
                for a in n.names:
                    if (submod, a.name) not in sigs:
                        issues.append(f"from elfx.{submod} import {a.name}: NOT FOUND in elfx.{submod}")
                    name_map[a.asname or a.name] = ("FUNC", submod, a.name)

    for n in ast.walk(tree):
        if not isinstance(n, ast.Call):
            continue
        key = None
        fn = n.func
        if (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Attribute)
            and isinstance(fn.value.value, ast.Name)
            and fn.value.value.id == "elfx"
        ):
            mod = fn.value.attr
            key = (mod, fn.attr)
            if mod not in imported_mods:
                issues.append(f"Line {n.lineno}: elfx.{mod}.{fn.attr} called but 'elfx.{mod}' is never imported")
        elif (
            isinstance(fn, ast.Attribute)
            and isinstance(fn.value, ast.Name)
            and name_map.get(fn.value.id, (None,))[0] == "MOD"
        ):
            key = (name_map[fn.value.id][1], fn.attr)
        elif isinstance(fn, ast.Name) and name_map.get(fn.id, (None,))[0] == "FUNC":
            key = name_map[fn.id][1:]

        if key is None:
            continue
        if key not in sigs:
            issues.append(f"Line {n.lineno}: elfx.{key[0]}.{key[1]} DOES NOT EXIST in elfx")
            continue

        params, varkw, varargs, nreq = sigs[key]
        if params is None:
            continue
        kws = [k.arg for k in n.keywords if k.arg]
        bad_kws = [k for k in kws if k not in params] if not varkw else []
        if bad_kws:
            issues.append(f"Line {n.lineno}: elfx.{key[0]}.{key[1]} unknown kwargs {bad_kws} (valid: {params})")
        npos = len(n.args)
        if not varargs and npos > len(params):
            issues.append(f"Line {n.lineno}: elfx.{key[0]}.{key[1]} too many positional arguments ({npos} > {len(params)})")
        provided = set(params[:npos]) | set(kws)
        missing = [p for p in params[:nreq] if p not in provided]
        if missing and not any(isinstance(a, ast.Starred) for a in n.args) and not any(k.arg is None for k in n.keywords):
            issues.append(f"Line {n.lineno}: elfx.{key[0]}.{key[1]} missing required parameter(s): {missing}")

    return issues


class TestNotebookApiImports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sigs = extract_elfx_signatures()

    def test_all_revision_notebooks_import_submodules_and_match_api(self):
        """Every revision notebook must import the elfx submodules it accesses and call valid signatures."""
        all_failures: Dict[str, List[str]] = {}
        for nb_name in REVISION_NOTEBOOKS:
            nb_path = REPO_ROOT / "notebooks" / nb_name
            self.assertTrue(nb_path.is_file(), f"Revision notebook {nb_name} does not exist")
            issues = check_notebook_api_calls(nb_path, self.sigs)
            if issues:
                all_failures[nb_name] = issues

        if all_failures:
            error_report = "\n".join(
                f"[{nb_name}]:\n  " + "\n  ".join(issues)
                for nb_name, issues in all_failures.items()
            )
            self.fail(f"Static AST API import check failed with {len(all_failures)} notebook(s) having issues:\n{error_report}")


if __name__ == "__main__":
    unittest.main()
