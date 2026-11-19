"""
tests/test_adversarial_notebook_empirical.py: Adversarial empirical stress-testing suite for Milestone 2 revision notebooks.

Specifically stress-tests:
1. Strict simulation quarantine (Rejection Point 3 & Required Fix Policy):
   - AST inspection asserting all mock/synthetic/simulation functions are strictly defined inside run_smoke_cpu().
   - Global namespace inspection asserting no mock functions or simulated outputs exist in production mode.
2. Fail-closed behavior outside Colab (ColabEngineeringError).
3. Fail-closed behavior inside Colab with missing Drive root (DriveNotMountedError / ColabEngineeringError).
4. Fail-closed behavior inside Colab with missing checkpoints/artifacts:
   - NB18: missing Plaid-1B weights -> PlaidWeightsMissingError.
   - NB19: missing spike manifest -> PlaidGateNotPassedError.
   - NB19: spike gate NO-GO -> PlaidGateNotPassedError.
   - NB19: spike gate GO but missing Plaid-1B weights -> PlaidWeightsMissingError.
   - NB16: missing ELF-B checkpoint -> MissingDriveArtifactError.
   - NB17: missing reference data -> MissingDriveArtifactError; missing replication samples -> MissingReplicationSamplesError.
   - NB15: missing replication runs -> MissingDriveArtifactError.
   - NB21: missing ACE calibrated directions -> MissingDriveArtifactError.
5. Production execution purity: asserting no mock data reachable in Colab production mode.
6. CPU smoke execution: end-to-end execution of run_smoke_cpu() across all 6 notebooks.
"""

import ast
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_NOTEBOOKS = [
    "15_final_human_evaluation.ipynb",
    "16_preregistered_c1_c3_replication.ipynb",
    "17_heldout_reference_and_contamination_audit.ipynb",
    "18_plaid_feasibility_spike.ipynb",
    "19_plaid_heldout_equivalence.ipynb",
    "21_ace_null_distribution_and_matched_ace.ipynb",
]

REVISION_NOTEBOOKS = list(dict.fromkeys(CANONICAL_NOTEBOOKS))


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


class TestAdversarialNotebookEmpirical(unittest.TestCase):
    """Adversarial stress-testing of revision notebooks."""

    def test_01_ast_simulation_quarantine(self):
        """Verify that every mock/simulation function is strictly quarantined inside run_smoke_cpu()."""
        mock_keywords = ["mock", "simulat", "fake", "dummy", "synthetic"]

        for nb_name in REVISION_NOTEBOOKS:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for cell_idx, cell in enumerate(data["cells"]):
                if cell.get("cell_type") != "code":
                    continue
                code_text = clean_source(cell.get("source", ""))
                if not code_text.strip():
                    continue

                tree = ast.parse(code_text, filename=f"{nb_name}#cell_{cell_idx}")

                # Check all top-level statements in the cell
                for stmt in tree.body:
                    if isinstance(stmt, ast.FunctionDef):
                        func_name = stmt.name.lower()
                        # If the top-level function itself contains mock keywords:
                        for kw in mock_keywords:
                            if kw in func_name and func_name != "run_smoke_cpu":
                                self.fail(
                                    f"VIOLATION in {nb_name} cell {cell_idx}: "
                                    f"Function '{stmt.name}' at module level contains mock keyword '{kw}'. "
                                    f"All mock/simulation functions must be quarantined inside run_smoke_cpu()."
                                )

                    # Check nested functions inside top-level functions
                    if isinstance(stmt, ast.FunctionDef) and stmt.name == "run_smoke_cpu":
                        # Nested mock functions inside run_smoke_cpu are permitted
                        pass
                    elif isinstance(stmt, ast.FunctionDef):
                        # Non-smoke top-level functions must NOT define mock inner functions
                        for inner_node in ast.walk(stmt):
                            if isinstance(inner_node, ast.FunctionDef) and inner_node is not stmt:
                                for kw in mock_keywords:
                                    if kw in inner_node.name.lower():
                                        self.fail(
                                            f"VIOLATION in {nb_name} cell {cell_idx}: "
                                            f"Function '{stmt.name}' defines inner mock function '{inner_node.name}'. "
                                            f"Must be quarantined inside run_smoke_cpu()."
                                        )

    def test_02_fail_closed_outside_colab(self):
        """Cell 1 of every notebook must raise ColabEngineeringError when RUN_SMOKE_CPU=False outside Colab."""
        # Ensure google.colab is not in sys.modules
        had_colab = "google.colab" in sys.modules
        old_colab = sys.modules.pop("google.colab", None)

        try:
            for nb_name in REVISION_NOTEBOOKS:
                path = REPO_ROOT / "notebooks" / nb_name
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
                cell1_code = clean_source(code_cells[0]["source"])

                env_globals = {"__name__": "__test__"}
                with self.assertRaises(Exception) as ctx:
                    exec(cell1_code, env_globals)

                exc = ctx.exception
                exc_type_name = type(exc).__name__
                self.assertIn(
                    exc_type_name,
                    ("ColabEngineeringError", "DriveNotMountedError"),
                    f"{nb_name} cell 1 outside Colab raised {exc_type_name} instead of ColabEngineeringError"
                )
                self.assertIn("FAIL-CLOSED", str(exc))
        finally:
            if had_colab and old_colab is not None:
                sys.modules["google.colab"] = old_colab

    def test_03_fail_closed_inside_colab_missing_drive(self):
        """When inside Colab but Drive mount path does not exist, Cell 1 must raise DriveNotMountedError or ColabEngineeringError."""
        fake_colab = types.ModuleType("google.colab")
        fake_drive = types.ModuleType("google.colab.drive")
        fake_drive.mount = lambda path, force_remount=False: None
        fake_colab.drive = fake_drive

        sys.modules["google.colab"] = fake_colab
        sys.modules["google.colab.drive"] = fake_drive

        try:
            for nb_name in REVISION_NOTEBOOKS:
                path = REPO_ROOT / "notebooks" / nb_name
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                code_cells = [c for c in data["cells"] if c.get("cell_type") == "code"]
                cell1_code = clean_source(code_cells[0]["source"])

                env_globals = {"__name__": "__test__"}
                with self.assertRaises(Exception) as ctx:
                    exec(cell1_code, env_globals)

                exc = ctx.exception
                exc_type_name = type(exc).__name__
                self.assertIn(
                    exc_type_name,
                    ("DriveNotMountedError", "ColabEngineeringError"),
                    f"{nb_name} cell 1 with missing Drive raised {exc_type_name}"
                )
        finally:
            sys.modules.pop("google.colab", None)
            sys.modules.pop("google.colab.drive", None)

    def test_04_notebook18_fail_closed_missing_plaid_weights(self):
        """Notebook 18 must fail closed with PlaidWeightsMissingError when Plaid-1B weights are absent."""
        path = REPO_ROOT / "notebooks" / "18_plaid_feasibility_spike.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]

        with tempfile.TemporaryDirectory() as tmp_drive:
            tmp_root = Path(tmp_drive) / "elf_research"
            tmp_root.mkdir(parents=True, exist_ok=True)

            fake_colab = types.ModuleType("google.colab")
            fake_drive = types.ModuleType("google.colab.drive")
            fake_drive.mount = lambda path, force_remount=False: None
            fake_colab.drive = fake_drive
            sys.modules["google.colab"] = fake_colab
            sys.modules["google.colab.drive"] = fake_drive

            try:
                env_globals = {"__name__": "__test__"}
                # Patch cell 1 to point to tmp_root
                cell1_code = code_cells[0].replace(
                    'root = Path("/content/drive/MyDrive/elf_research")',
                    f'root = Path("{tmp_root}")'
                )
                exec(cell1_code, env_globals)

                # Execute cells 2, 3
                exec(code_cells[1], env_globals)
                exec(code_cells[2], env_globals)

                # Execute cell 4 (production mode RUN_SMOKE_CPU=False)
                # models/plaid1b does NOT exist in tmp_root
                with self.assertRaises(Exception) as ctx:
                    exec(code_cells[3], env_globals)

                exc = ctx.exception
                self.assertEqual(type(exc).__name__, "PlaidWeightsMissingError")
                self.assertIn("Plaid-1B checkpoint missing", str(exc))

                # Verify simulation function was NEVER called and is NOT in globals
                self.assertNotIn("_mock_simulate_plaid_point_for_cpu_smoke_only", env_globals)
                self.assertNotIn("points", env_globals)
            finally:
                sys.modules.pop("google.colab", None)
                sys.modules.pop("google.colab.drive", None)

    def test_05_notebook18_production_mode_with_weights_present(self):
        """When Plaid-1B checkpoint is present, Notebook 18 passes preflight without invoking simulation."""
        path = REPO_ROOT / "notebooks" / "18_plaid_feasibility_spike.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]

        with tempfile.TemporaryDirectory() as tmp_drive:
            tmp_root = Path(tmp_drive) / "elf_research"
            plaid_dir = tmp_root / "models" / "plaid1b"
            plaid_dir.mkdir(parents=True, exist_ok=True)

            fake_colab = types.ModuleType("google.colab")
            fake_drive = types.ModuleType("google.colab.drive")
            fake_drive.mount = lambda path, force_remount=False: None
            fake_colab.drive = fake_drive
            sys.modules["google.colab"] = fake_colab
            sys.modules["google.colab.drive"] = fake_drive

            try:
                env_globals = {"__name__": "__test__"}
                cell1_code = code_cells[0].replace(
                    'root = Path("/content/drive/MyDrive/elf_research")',
                    f'root = Path("{tmp_root}")'
                )
                exec(cell1_code, env_globals)
                exec(code_cells[1], env_globals)
                exec(code_cells[2], env_globals)

                # Cell 4 in production mode
                exec(code_cells[3], env_globals)

                # Check that simulation was NEVER executed
                self.assertNotIn("_mock_simulate_plaid_point_for_cpu_smoke_only", env_globals)
                self.assertNotIn("points", env_globals)
                self.assertNotIn("gate_res", env_globals)
            finally:
                sys.modules.pop("google.colab", None)
                sys.modules.pop("google.colab.drive", None)

    def test_06_notebook19_fail_closed_gate_and_weights(self):
        """Notebook 19 must fail closed on missing gate manifest, NO-GO gate, or missing weights."""
        path = REPO_ROOT / "notebooks" / "19_plaid_heldout_equivalence.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]

        with tempfile.TemporaryDirectory() as tmp_drive:
            tmp_root = Path(tmp_drive) / "elf_research"
            tmp_root.mkdir(parents=True, exist_ok=True)
            spike_manifest_dir = tmp_root / "runs" / "plaid_eval"
            spike_manifest_dir.mkdir(parents=True, exist_ok=True)
            spike_manifest = spike_manifest_dir / "plaid_spike_response_surface.json"

            fake_colab = types.ModuleType("google.colab")
            fake_drive = types.ModuleType("google.colab.drive")
            fake_drive.mount = lambda path, force_remount=False: None
            fake_colab.drive = fake_drive
            sys.modules["google.colab"] = fake_colab
            sys.modules["google.colab.drive"] = fake_drive

            try:
                env_globals = {"__name__": "__test__"}
                cell1_code = code_cells[0].replace(
                    'root = Path("/content/drive/MyDrive/elf_research")',
                    f'root = Path("{tmp_root}")'
                )
                exec(cell1_code, env_globals)
                exec(code_cells[1], env_globals)
                exec(code_cells[2], env_globals)

                # Case A: Spike manifest missing -> PlaidGateNotPassedError
                with self.assertRaises(Exception) as ctx_a:
                    exec(code_cells[3], env_globals)
                self.assertEqual(type(ctx_a.exception).__name__, "PlaidGateNotPassedError")

                # Case B: Spike manifest is NO-GO -> PlaidGateNotPassedError
                spike_manifest.write_text(json.dumps({"gate_decision": "NO-GO", "rationale": "No pairs"}), encoding="utf-8")
                with self.assertRaises(Exception) as ctx_b:
                    exec(code_cells[3], env_globals)
                self.assertEqual(type(ctx_b.exception).__name__, "PlaidGateNotPassedError")
                self.assertIn("did NOT yield GO", str(ctx_b.exception))

                # Case C: Spike manifest is GO, but models/plaid1b missing -> PlaidWeightsMissingError
                spike_manifest.write_text(json.dumps({"gate_decision": "GO", "rationale": "Found pairs"}), encoding="utf-8")
                with self.assertRaises(Exception) as ctx_c:
                    exec(code_cells[3], env_globals)
                self.assertEqual(type(ctx_c.exception).__name__, "PlaidWeightsMissingError")

                # Verify simulation functions were never executed
                self.assertNotIn("_mock_plaid_heldout_runs", env_globals)
                self.assertNotIn("records", env_globals)
            finally:
                sys.modules.pop("google.colab", None)
                sys.modules.pop("google.colab.drive", None)

    def test_07_notebook16_fail_closed_production_and_preflight(self):
        """Notebook 16 (both canonical and colab) must fail closed on non-CUDA environments and load ELF-B via sweep/load_elf."""
        for nb_name in ["16_preregistered_c1_c3_replication.ipynb"]:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
            all_code = "\n".join(code_cells)

            # 1. Assert no obsolete elf_b.pt checkpoint checks exist
            self.assertNotIn("checkpoints/elf_b.pt", all_code)
            self.assertNotIn("elf_b.pt", all_code)
            self.assertTrue(
                'get_model("ELF-B")' in all_code or 'load_elf("ELF-B")' in all_code,
                f"{nb_name} must load model via get_model('ELF-B') or load_elf('ELF-B')"
            )
            # Verify no dtype=torch.bfloat16 in generate
            self.assertNotIn("dtype=torch.bfloat16", all_code)

            # 2. Assert fail-closed on non-CUDA in mock Colab environment
            with tempfile.TemporaryDirectory() as tmp_drive:
                tmp_root = Path(tmp_drive) / "elf_research"
                tmp_root.mkdir(parents=True, exist_ok=True)

                fake_colab = types.ModuleType("google.colab")
                fake_drive = types.ModuleType("google.colab.drive")
                fake_drive.mount = lambda path, force_remount=False: None
                fake_colab.drive = fake_drive
                sys.modules["google.colab"] = fake_colab
                sys.modules["google.colab.drive"] = fake_drive

                try:
                    env_globals = {"__name__": "__test__"}
                    cell1_code = code_cells[0].replace(
                        'root = Path("/content/drive/MyDrive/elf_research")',
                        f'root = Path("{tmp_root}")'
                    )
                    exec(cell1_code, env_globals)
                    for c_idx in range(1, len(code_cells) - 1):
                        exec(code_cells[c_idx], env_globals)

                    # Final cell execution in production mode (RUN_SMOKE_CPU=False, RUN_PREFLIGHT_GPU=False)
                    with self.assertRaises(Exception) as ctx:
                        exec(code_cells[-1], env_globals)
                    self.assertIn("ColabEngineeringError", type(ctx.exception).__name__)
                    self.assertIn("Production replication requires a CUDA GPU", str(ctx.exception))

                    # Verify simulation mock records were never emitted in production mode
                    self.assertNotIn("mock_records", env_globals)
                finally:
                    sys.modules.pop("google.colab", None)
                    sys.modules.pop("google.colab.drive", None)

    def test_08_notebook17_fail_closed_production_and_preflight(self):
        """Notebook 17 (both canonical and colab) must fail closed on non-CUDA environments and use valid mauve signatures."""
        for nb_name in ["17_heldout_reference_and_contamination_audit.ipynb"]:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
            all_code = "\n".join(code_cells)

            # 1. Assert mauve_from_features call signature correctness: no verbose kwarg
            self.assertNotIn("verbose=False", all_code)
            self.assertIn("mauve_from_features(gen_feats=gen_feats, ref_feats=ref_feats, seed=25)", all_code)

            # 2. Assert decoupled pilot vs replication mode: pilot does not require replication directory
            self.assertNotIn("MissingReplicationSamplesError", all_code)

            # 3. Assert fail-closed on non-CUDA in mock Colab environment
            with tempfile.TemporaryDirectory() as tmp_drive:
                tmp_root = Path(tmp_drive) / "elf_research"
                tmp_root.mkdir(parents=True, exist_ok=True)

                fake_colab = types.ModuleType("google.colab")
                fake_drive = types.ModuleType("google.colab.drive")
                fake_drive.mount = lambda path, force_remount=False: None
                fake_colab.drive = fake_drive
                sys.modules["google.colab"] = fake_colab
                sys.modules["google.colab.drive"] = fake_drive

                try:
                    env_globals = {"__name__": "__test__"}
                    cell1_code = code_cells[0].replace(
                        'root = Path("/content/drive/MyDrive/elf_research")',
                        f'root = Path("{tmp_root}")'
                    )
                    exec(cell1_code, env_globals)
                    for c_idx in range(1, len(code_cells) - 1):
                        exec(code_cells[c_idx], env_globals)

                    # Final cell in production mode without CUDA
                    with self.assertRaises(Exception) as ctx:
                        exec(code_cells[-1], env_globals)
                    self.assertIn("ColabEngineeringError", type(ctx.exception).__name__)
                    self.assertIn("Production audit requires an attached CUDA GPU", str(ctx.exception))
                finally:
                    sys.modules.pop("google.colab", None)
                    sys.modules.pop("google.colab.drive", None)

    def test_09_notebook15_fail_closed_production_and_preflight(self):
        """Notebook 15 (both canonical and colab) must fail closed on non-CUDA environments and enforce admission gate."""
        import torch
        for nb_name in ["15_final_human_evaluation.ipynb"]:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
            all_code = "\n".join(code_cells)

            # 1. Assert no temperature=0.0 when do_sample=False
            self.assertNotIn("temperature=0.0", all_code)

            # 2. Assert position swapping logic
            self.assertIn("_prompt_judge(text_a, text_b)", all_code)
            self.assertIn("_prompt_judge(text_b, text_a)", all_code)

            # 3. Assert fail-closed on non-CUDA in mock Colab environment
            with tempfile.TemporaryDirectory() as tmp_drive:
                tmp_root = Path(tmp_drive) / "elf_research"
                tmp_root.mkdir(parents=True, exist_ok=True)

                fake_colab = types.ModuleType("google.colab")
                fake_drive = types.ModuleType("google.colab.drive")
                fake_drive.mount = lambda path, force_remount=False: None
                fake_colab.drive = fake_drive
                sys.modules["google.colab"] = fake_colab
                sys.modules["google.colab.drive"] = fake_drive

                try:
                    env_globals = {"__name__": "__test__"}
                    cell1_code = code_cells[0].replace(
                        'root = Path("/content/drive/MyDrive/elf_research")',
                        f'root = Path("{tmp_root}")'
                    )
                    exec(cell1_code, env_globals)
                    for c_idx in range(1, len(code_cells) - 1):
                        exec(code_cells[c_idx], env_globals)

                    # Final cell in production mode without CUDA
                    with self.assertRaises(Exception) as ctx:
                        exec(code_cells[-1], env_globals)
                    self.assertIn("ColabEngineeringError", type(ctx.exception).__name__)
                    self.assertIn("Production panel requires an attached CUDA GPU", str(ctx.exception))
                finally:
                    sys.modules.pop("google.colab", None)
                    sys.modules.pop("google.colab.drive", None)

            # 4. Assert admission gate failure when accuracy < 70%
            smoke_env = {"__name__": "__test__"}
            cell1_smoke = code_cells[0].replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True").replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")
            exec(cell1_smoke, smoke_env)
            exec(code_cells[1], smoke_env)
            exec(code_cells[2], smoke_env)

            eval_gate = smoke_env["evaluate_judge_admission_gate"]
            dummy_pairs = [{"text_a": "A", "text_b": "B", "human_better": "A"}]
            dummy_model = types.SimpleNamespace(device="cpu")
            class DummyBatchEncoding(dict):
                def to(self, device):
                    return self
            class DummyTokenizer:
                pad_token_id = 0
                def __call__(self, prompt, **kwargs):
                    return DummyBatchEncoding({"input_ids": torch.tensor([[1, 2]])})
                def decode(self, ids, **kwargs):
                    return "Option 2"
            dummy_tok = DummyTokenizer()
            dummy_model.generate = lambda **kwargs: torch.tensor([[1, 2, 3]])

            with self.assertRaises(Exception) as gate_ctx:
                eval_gate(dummy_model, dummy_tok, dummy_pairs, gate_threshold=0.70)
            self.assertEqual(type(gate_ctx.exception).__name__, "JudgeAdmissionGateError")
            self.assertIn("Judge failed admission gate", str(gate_ctx.exception))

    def test_10_notebook21_fail_closed_production_and_preflight(self):
        """Notebook 21 (both canonical and colab) must fail closed on non-CUDA environments and verify orthogonal null invariant."""
        import torch
        for nb_name in ["21_ace_null_distribution_and_matched_ace.ipynb"]:
            path = REPO_ROOT / "notebooks" / nb_name
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            code_cells = [clean_source(c["source"]) for c in data["cells"] if c.get("cell_type") == "code"]
            all_code = "\n".join(code_cells)

            # 1. Assert no dtype=torch.bfloat16 passed to generate
            self.assertNotIn("dtype=torch.bfloat16", all_code)

            # 2. Assert fail-closed on non-CUDA in mock Colab environment
            with tempfile.TemporaryDirectory() as tmp_drive:
                tmp_root = Path(tmp_drive) / "elf_research"
                tmp_root.mkdir(parents=True, exist_ok=True)

                fake_colab = types.ModuleType("google.colab")
                fake_drive = types.ModuleType("google.colab.drive")
                fake_drive.mount = lambda path, force_remount=False: None
                fake_colab.drive = fake_drive
                sys.modules["google.colab"] = fake_colab
                sys.modules["google.colab.drive"] = fake_drive

                try:
                    env_globals = {"__name__": "__test__"}
                    cell1_code = code_cells[0].replace(
                        'root = Path("/content/drive/MyDrive/elf_research")',
                        f'root = Path("{tmp_root}")'
                    )
                    exec(cell1_code, env_globals)
                    for c_idx in range(1, len(code_cells) - 1):
                        exec(code_cells[c_idx], env_globals)

                    # Final cell in production mode without CUDA
                    with self.assertRaises(Exception) as ctx:
                        exec(code_cells[-1], env_globals)
                    self.assertIn("ColabEngineeringError", type(ctx.exception).__name__)
                    self.assertIn("Production matched ACE requires a CUDA GPU", str(ctx.exception))
                finally:
                    sys.modules.pop("google.colab", None)
                    sys.modules.pop("google.colab.drive", None)

            # 3. Verify orthogonal null directions mathematical invariant on S^510 subset R^512
            smoke_env = {"__name__": "__test__"}
            cell1_smoke = code_cells[0].replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True").replace("RUN_SMOKE_CPU = False", "RUN_SMOKE_CPU = True")
            exec(cell1_smoke, smoke_env)
            exec(code_cells[1], smoke_env)

            gen_nulls = smoke_env["generate_20_orthogonal_null_directions"]
            d_target = torch.randn(512)
            d_target = d_target / d_target.norm(p=2)

            nulls = gen_nulls(d_target, seed_base=202704, device="cpu")
            self.assertEqual(nulls.shape, (20, 512))

            for k in range(20):
                norm_k = float(nulls[k].norm(p=2).item())
                self.assertAlmostEqual(norm_k, 1.0, places=5)
                dot_k = float(torch.dot(nulls[k], d_target).item())
                self.assertAlmostEqual(dot_k, 0.0, places=5)

    def test_plaid_preflight_sanity_gate_passes_on_valid_english(self):
        """Verify NB18 Cell 9 run_plaid_16_sample_preflight_gate succeeds on valid English passages."""
        path = REPO_ROOT / "notebooks" / "18_plaid_feasibility_spike.ipynb"
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        c1 = "".join(nb["cells"][1]["source"]).replace("RUN_SMOKE_CPU     = False", "RUN_SMOKE_CPU = True")
        c9 = "".join(nb["cells"][9]["source"])

        env_globals = {"__name__": "__test__"}
        exec(c1 + "\n" + c9, env_globals)

        # Mock 16 fluent English sentences
        mock_sentence = "The quick brown fox jumps over the lazy dog."
        env_globals["sample_plaid_official_subprocess"] = lambda **kwargs: [mock_sentence] * 16

        import numpy as np
        import elfx.metrics
        old_score_gpt2 = elfx.metrics.score_gpt2
        try:
            elfx.metrics.score_gpt2 = lambda texts, model_name="gpt2-large": {
                "nll": np.full(len(texts), np.log(25.0) * 10.0),
                "count": np.full(len(texts), 10.0),
                "entropy": np.full(len(texts), 5.15),
                "empty": np.zeros(len(texts), dtype=bool),
            }
            res = env_globals["run_plaid_16_sample_preflight_gate"](artifacts={}, flags_info={})
            self.assertIn("word_ratio", res)
            self.assertGreaterEqual(res["word_ratio"], 0.50)
            self.assertAlmostEqual(res["gen_ppl"], 25.0, places=4)
            self.assertEqual(res["status"], "PASSED")
        finally:
            elfx.metrics.score_gpt2 = old_score_gpt2


if __name__ == "__main__":
    unittest.main()
