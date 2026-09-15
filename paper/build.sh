#!/bin/bash
# Build manuscript/main.tex (the per-section source of truth since 2026-09-13):
# pdflatex, bibtex, pdflatex x2. Reports errors, undefined refs, overflows.
export PATH="/c/Users/ongs6/AppData/Local/Programs/MiKTeX/miktex/bin/x64:$PATH"
cd "$(dirname "$0")/manuscript" || exit 1
B=main
pdflatex -interaction=nonstopmode $B.tex > /dev/null 2>&1
bibtex $B > /dev/null 2>&1
pdflatex -interaction=nonstopmode $B.tex > /dev/null 2>&1
pdflatex -interaction=nonstopmode $B.tex > /dev/null 2>&1
echo "=== hard errors ==="
grep -E "^!" $B.log | sort -u || echo "none"
echo "=== undefined refs/cites ==="
grep -oE "(Reference|Citation) .[^']+.' .* undefined" $B.log | sort -u | head -20
echo "(count: $(grep -cE '(Reference|Citation) .* undefined' $B.log))"
# A literal backslash in the pattern must be written as [\\] inside single
# quotes; the previous "\\hbox" form collapsed to \h, matched nothing, and
# silently reported 0 while boxes up to 162pt overflowed the margin.
echo "=== overfull boxes over 10pt ==="
grep -E 'Overfull [\\]hbox \(([1-9][0-9]|[0-9]{3,})\.' $B.log || echo "none"
echo "=== output ==="
grep -E "Output written" $B.log | tail -1
