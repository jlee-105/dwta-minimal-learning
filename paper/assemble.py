"""Assemble manuscript_cor_merged.tex from the v2 drafts plus the sections
carried over unchanged from BReRLA_manuscript.tex.

Run after editing any draft; it regenerates the merged file from scratch so
fixes live in the drafts rather than in the assembled output.
"""
import re

SRC = open("BReRLA_manuscript.tex", encoding="utf-8").read().split("\n")


def between(start, end, include_start=True):
    """Lines of the shipped manuscript from the line equal to `start` up to,
    not including, the line equal to `end`. Both must occur exactly once at
    line start.

    Replaces the old grab(a, b) line-number ranges: editing an equation inside
    a carried-over block shifted every later line, which silently truncated one
    block mid-equation and pulled a duplicate \\section header into another."""
    lines = [l.rstrip("\r") for l in SRC]
    s = [i for i, l in enumerate(lines) if l == start]
    e = [i for i, l in enumerate(lines) if l == end]
    assert len(s) == 1 and len(e) == 1, "marker not unique: %r / %r" % (start, end)
    a = s[0] if include_start else s[0] + 1
    return "\n".join(lines[a:e[0]]).rstrip() + "\n"


def load(p):
    return open(p, encoding="utf-8").read()


parts = []

# ---- preamble + frontmatter from the skeleton, up to the assembly comment
skel = load("manuscript_cor.tex")
parts.append(skel.split("%% ============================================================\n%% ASSEMBLY ORDER")[0])

# ---- abstract body spliced into the skeleton's empty abstract environment
rel = load("draft_related_abstract_v2.tex")
abstract = rel.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
parts[0] = parts[0].replace(
    "\\begin{abstract}\n"
    "%% SOURCE: draft_related_abstract_v2.tex -- paste the abstract body here once\n"
    "%% the [[N]] placeholders are resolved by the multi-seed results.\n"
    "\\end{abstract}",
    "\\begin{abstract}" + abstract + "\\end{abstract}")

# ---- Introduction: fully self-contained in the draft as of the 2026-09-13
#      line-by-line editing pass (MOVES 1-3 are now embedded in the file
#      too, not carried over from BReRLA_manuscript.tex 699-708 as before).
parts.append("\\section{Introduction}\n\\label{sec:introduction}\n")
parts.append(load("draft_intro_moves456_v2.tex"))

# ---- Related Work. Rewritten 2026-09-13 as one continuous section with no
#      subsections; the draft now carries the header, the label and the
#      positioning table, so nothing is carried over from the shipped file.
parts.append("\n" + rel.split("\\end{abstract}")[1])

# ---- Problem Formulation. Rewritten 2026-09-13 into its own draft; no longer
#      carried over from BReRLA_manuscript.tex 797-902.
parts.append("\n" + load("draft_problem_v2.tex"))

# ---- MDP + Method. Splice the real encoding subsection over its pointer; the
#      carried-over block at 1014 already emits the subsection header and label.
mdpm = load("draft_mdp_method_v2.tex")
pointer = ("\\subsection{Heterogeneous Graph Encoding}\n"
           "\\label{sec:encoding}\n"
           "%% UNCHANGED from the shipped manuscript -- Eqs. \\eqref{eq:gnn-proj}--\\eqref{eq:gnn-node}\n"
           "%% still apply verbatim. Retained here as a pointer only; copy the existing\n"
           "%% subsection body in when assembling the full section.")
assert pointer in mdpm, "encoding pointer not found -- draft changed?"
mdpm = mdpm.replace(pointer, between("\\subsection{Heterogeneous Graph Encoding}",
                                       "\\subsection{Weapon-to-Weapon Communication}"))

# The Method's subsections follow the order the pipeline actually runs in:
# encode, decide fire/hold, assign targets, improve, train. The auction draft
# is spliced in at the third position. prop:auction-half lives in the theory
# section with the other three propositions.
ls_head = "\\subsection{Local Search Over the Completed Schedule}"
assert ls_head in mdpm, "local-search subsection header not found -- draft changed?"
mdpm = mdpm.replace(ls_head, load("draft_auction_v2.tex").strip() + "\n\n" + ls_head, 1)
parts.append("\n" + mdpm)

# ---- Theory
parts.append("\n" + load("draft_theory_v2.tex"))

# ---- Experiments. Splice the carried-over data + hyperparameter subsections.
exp = load("draft_experiments_v2.tex")
carry = ("%% Instance distributions (Table~\\ref{tab:parameter_distributions}), the twelve\n"
         "%% held-out configurations (Table~\\ref{tab:problem_instance_scales}) and the\n"
         "%% training curriculum carry over unchanged from the shipped manuscript.")
assert carry in exp, "experiments carry-over marker not found -- draft changed?"
exp = exp.replace(carry, between("\\subsection{Training and Test Data}",
                                 "\\subsection{Performance Measures and Baselines}",
                                 include_start=False))
parts.append("\n" + exp)

# ---- Conclusion
parts.append("\n" + load("draft_conclusion_v2.tex"))

# ---- Back matter from the skeleton (declarations + bibliography), then the
#      appendix, which must follow \bibliography for elsarticle numbering.
BACK_MARKER = "%% ============================================================\n%% CRediT statement."
assert BACK_MARKER in skel, "back-matter marker not found in manuscript_cor.tex"
back = skel[skel.index(BACK_MARKER):]
appendix = load("draft_appendix_v2.tex")
back = back.replace("\\end{document}", appendix + "\n\\end{document}")
parts.append("\n" + back)

out = "\n".join(parts)
open("manuscript_cor_merged.tex", "w", encoding="utf-8").write(out)

live = re.sub(r"(?<!\\)%.*", "", out)
labels = re.findall(r"\\label\{([^}]+)\}", live)
dupes = sorted({l for l in labels if labels.count(l) > 1})
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", live))
missing = sorted(refs - set(labels))

print("wrote manuscript_cor_merged.tex: %d lines" % (out.count("\n") + 1))
print("duplicate labels : %s" % (dupes or "none"))
print("dangling refs    : %s" % (missing or "none"))
