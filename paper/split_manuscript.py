"""Split manuscript_cor_merged.tex into one .tex file per section under
manuscript/, plus main.tex that \\input's them in order. The folder is
self-contained (bib + figures copied) so it can be uploaded to Overleaf as is.

From 2026-09-13 the files in manuscript/ are the source of truth. Edit them
directly; do not re-run assemble.py + this script, or those edits are lost.
"""
import io
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "manuscript")
L = io.open(os.path.join(HERE, "manuscript_cor_merged.tex"), encoding="utf-8").read().split("\n")

SECTIONS = [
    ("\\section{Introduction}", "introduction"),
    ("\\section{Related Work}", "related_works"),
    ("\\section{Problem Formulation}", "problem_definition"),
    ("\\section{Markov Decision Process Formulation}", "MDP"),
    ("\\section{Proposed Method}", "method"),
    ("\\section{Theoretical Analysis}", "theoretical_analysis"),
    ("\\section{Experimental Results}", "Experiment"),
    ("\\section{Conclusion}", "conclusion"),
]


def idx(prefix):
    hits = [i for i, l in enumerate(L) if l.startswith(prefix)]
    assert len(hits) == 1, (prefix, hits)
    return hits[0]


def trim(lines):
    """Drop trailing blank lines and the %%-comment header block that belongs
    to the NEXT section's draft file."""
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("%%")):
        lines.pop()
    return lines


starts = [idx(p) for p, _ in SECTIONS]
back = next(i for i, l in enumerate(L) if "CRediT statement" in l) - 1   # the %% ==== rule above it
bib = idx("\\bibliography{")
app = idx("\\section{Evaluation traces}")
end = idx("\\end{document}")

os.makedirs(OUT, exist_ok=True)
bounds = starts + [back]
for k, (_, name) in enumerate(SECTIONS):
    body = trim(L[bounds[k]:bounds[k + 1]])
    io.open(os.path.join(OUT, name + ".tex"), "w", encoding="utf-8").write("\n".join(body) + "\n")
    print("wrote %-24s %4d lines" % (name + ".tex", len(body)))

appx = trim(L[app:end])
io.open(os.path.join(OUT, "appendix.tex"), "w", encoding="utf-8").write("\n".join(appx) + "\n")
print("wrote %-24s %4d lines" % ("appendix.tex", len(appx)))

head = trim(L[:starts[0]])
main = (head
        + ["", "%% \\input rather than \\include: \\include forces a page break before and",
           "%% after every section, which a journal manuscript should not have."]
        + ["\\input{%s}" % name for _, name in SECTIONS]
        + [""] + L[back:bib + 1]
        + ["", "\\appendix", "\\input{appendix}", "", "\\end{document}", ""])
io.open(os.path.join(OUT, "main.tex"), "w", encoding="utf-8").write("\n".join(main))
print("wrote main.tex")

shutil.copy(os.path.join(HERE, "references.bib"), OUT)
os.makedirs(os.path.join(OUT, "figure"), exist_ok=True)
for f in ("Overall.png", "Training.png"):
    shutil.copy(os.path.join(HERE, "figure", f), os.path.join(OUT, "figure", f))
print("copied references.bib and figures")
