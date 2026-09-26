"""Static checks for paper/main.tex (NOT a substitute for compiling).

Checks: citation keys exist in references.bib; \\ref/\\cref targets are defined;
\\num* macros are defined in generated/numbers.tex; \\input files exist;
begin/end environments balance; brace balance per file; anonymisation
patterns; rough word count of the main body (before \\appendix).

Usage: python3 scripts/check_tex.py
Exit code 0 if no errors; warnings are printed but do not fail.
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")


def read(p):
    with open(p) as f:
        return f.read()


def strip_comments(s):
    return re.sub(r"(?<!\\)%.*", "", s)


def expand_inputs(s, base):
    def rep(m):
        name = m.group(1)
        path = os.path.join(base, name if name.endswith(".tex") else name + ".tex")
        if not os.path.exists(path):
            return f"<<MISSING INPUT {name}>>"
        return strip_comments(read(path))
    return re.sub(r"\\input\{([^}]+)\}", rep, s)


def main():
    errors, warnings = [], []
    main_src = strip_comments(read(os.path.join(PAPER, "main.tex")))
    full = expand_inputs(main_src, PAPER)
    for m in re.finditer(r"<<MISSING INPUT ([^>]+)>>", full):
        errors.append(f"missing \\input file: {m.group(1)}")

    bib = read(os.path.join(PAPER, "references.bib"))
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = set()
    for m in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", full):
        cited.update(k.strip() for k in m.group(1).split(","))
    for k in sorted(cited - keys):
        errors.append(f"citation key not in references.bib: {k}")
    unused = sorted(keys - cited)
    if unused:
        warnings.append(f"bib entries not cited (harmless): {', '.join(unused)}")

    labels = set(re.findall(r"\\label\{([^}]+)\}", full))
    refs = set()
    for m in re.finditer(r"\\(?:c|C)?ref\{([^}]+)\}", full):
        refs.update(r.strip() for r in m.group(1).split(","))
    for r in sorted(refs - labels):
        errors.append(f"undefined reference: {r}")

    nums = read(os.path.join(PAPER, "generated", "numbers.tex"))
    defined = set(re.findall(r"\\newcommand\{\\(num\w+)\}", nums))
    used = set(re.findall(r"\\(num[A-Za-z]+)", full))
    for u in sorted(used - defined):
        errors.append(f"undefined numeric macro: \\{u}")

    begins = re.findall(r"\\begin\{([^}]+)\}", full)
    ends = re.findall(r"\\end\{([^}]+)\}", full)
    for env in set(begins) | set(ends):
        if begins.count(env) != ends.count(env):
            errors.append(f"environment imbalance: {env} begin={begins.count(env)} end={ends.count(env)}")

    for fname in ["main.tex", "fig_numerics.tex"] + [os.path.join("generated", f) for f in os.listdir(os.path.join(PAPER, "generated")) if f.endswith(".tex")]:
        s = strip_comments(read(os.path.join(PAPER, fname))).replace("\\{", "").replace("\\}", "")
        if s.count("{") != s.count("}"):
            errors.append(f"brace imbalance in {fname}: {{={s.count('{')} }}={s.count('}')}")

    anon_patterns = [r"donggil", r"pusan", r"github\.com", r"claude", r"anthropic", r"@pusan", r"Subset-Compatibility"]
    raw_main = read(os.path.join(PAPER, "main.tex"))
    body_no_comments = strip_comments(raw_main)
    for pat in anon_patterns:
        if re.search(pat, body_no_comments, re.I):
            errors.append(f"anonymisation pattern found in main.tex text: {pat}")

    todos = re.findall(r"\\todo\{([^}]*)", full)
    body = full.split("\\appendix")[0]
    body = body.split("\\begin{document}")[1] if "\\begin{document}" in body else body
    body_text = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    body_text = re.sub(r"[{}$\\&_^~]", " ", body_text)
    words = len([w for w in body_text.split() if re.search(r"[A-Za-z]", w)])

    print(f"citations: {len(cited)} used / {len(keys)} in bib")
    print(f"labels: {len(labels)}; references used: {len(refs)}")
    print(f"numeric macros used: {len(used)} / defined: {len(defined)}")
    print(f"TODO markers: {len(todos)}")
    for t in todos:
        print(f"  - {t.split(':')[0]}")
    print(f"approx. words before appendix (incl. tables/captions): {words}")
    for w in warnings:
        print("WARNING:", w)
    for e in errors:
        print("ERROR:", e)
    print("STATIC_CHECK:", "PASS" if not errors else "FAIL", "(not a compile)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
