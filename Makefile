PYTHON ?= $(shell if [ -x ./ELF/.venv/bin/python3 ]; then echo ./ELF/.venv/bin/python3; elif [ -x ../ELF/.venv/bin/python3 ]; then echo ../ELF/.venv/bin/python3; else echo python3; fi)
PDFLATEX ?= $(shell if command -v pdflatex >/dev/null 2>&1; then echo pdflatex; elif [ -x /Library/TeX/texbin/pdflatex ]; then echo /Library/TeX/texbin/pdflatex; else echo pdflatex; fi)
BIBTEX ?= $(shell if command -v bibtex >/dev/null 2>&1; then echo bibtex; elif [ -x /Library/TeX/texbin/bibtex ]; then echo /Library/TeX/texbin/bibtex; else echo bibtex; fi)

.PHONY: all analysis test check pdf

all: analysis test check

analysis:
	$(PYTHON) analysis/statistical_reanalysis.py

test:
	$(PYTHON) tests/run_tests.py
	$(PYTHON) -m unittest discover -s tests

check:
	$(PYTHON) analysis/check_manuscript.py

pdf:
	$(PDFLATEX) -interaction=nonstopmode main.tex
	$(BIBTEX) main
	$(PDFLATEX) -interaction=nonstopmode main.tex
	$(PDFLATEX) -interaction=nonstopmode main.tex
