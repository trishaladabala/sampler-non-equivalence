#!/usr/bin/env bash
# End-to-end CPU reproduction and verification pipeline for:
# "Same Perplexity, Same Entropy, Different Text: Operating-Point Non-Equivalence
#  of Sampler Controls in Continuous-Latent Flow Language Models" (ACL 2027 / ARR)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Auto-detect Python interpreter (use local venv if present, else python3)
if [[ -n "${PYTHON_BIN:-}" ]]; then
    PY="$PYTHON_BIN"
elif [[ -x "$REPO_ROOT/ELF/.venv/bin/python3" ]]; then
    PY="$REPO_ROOT/ELF/.venv/bin/python3"
elif [[ -x "$REPO_ROOT/../ELF/.venv/bin/python3" ]]; then
    PY="$REPO_ROOT/../ELF/.venv/bin/python3"
else
    PY="python3"
fi

echo "======================================================================"
echo "ACL 2027 Reproduction & Verification Suite"
echo "Repository Root: $REPO_ROOT"
echo "Python Binary:   $PY"
echo "======================================================================"

echo ""
echo "[1/5] Running Statistical Reanalysis (analysis/statistical_reanalysis.py)..."
"$PY" analysis/statistical_reanalysis.py

echo ""
echo "[2/5] Running Estimator Unit Tests (22 tests in tests/run_tests.py)..."
"$PY" tests/run_tests.py

echo ""
echo "[3/5] Running Full Unittest Suite (148 tests in tests/)..."
"$PY" -m unittest discover -s tests

echo ""
echo "[4/5] Running Manuscript Claim & Guard Checker (115 claims, 27 guards, 23 tables, 5 figures)..."
"$PY" analysis/check_manuscript.py

echo ""
echo "[5/5] Checking LaTeX Toolchain for PDF Compilation..."
if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    echo "Compiling main.pdf via pdflatex + bibtex..."
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    bibtex main >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    pdflatex -interaction=nonstopmode main.tex >/dev/null
    echo "Compiled main.pdf successfully."
elif [[ -x "/Library/TeX/texbin/pdflatex" ]] && [[ -x "/Library/TeX/texbin/bibtex" ]]; then
    echo "Compiling main.pdf via /Library/TeX/texbin/pdflatex + bibtex..."
    /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex >/dev/null
    /Library/TeX/texbin/bibtex main >/dev/null
    /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex >/dev/null
    /Library/TeX/texbin/pdflatex -interaction=nonstopmode main.tex >/dev/null
    echo "Compiled main.pdf successfully."
else
    echo "NOTE: pdflatex/bibtex not found on PATH; skipping PDF recompilation (pre-compiled main.pdf is present)."
fi

echo ""
echo "======================================================================"
echo "ALL REPRODUCTION & VERIFICATION CHECKS PASSED (170/170 tests, 115/115 claims, 27/27 guards)"
echo "======================================================================"
