# Building the manuscript (built in round 6; see paper/build_log_summary.json)

**Status:** COMPILED locally in round 6 (`paper/main.pdf`; details in
`paper/build_log_summary.json`).

- Built with a minimal TeX Live 2026 installed under a user-delivered
  limited approval; the official ICML 2026 style was fetched by
  `fetch_style.sh` (SHA-256 verified) and used unmodified in review mode.
- 19 pages; the conclusion's last sentence is on page 8 (measured with
  `\pdfsavepos`); no overfull boxes; no undefined references.
- The pages were NOT rasterized or viewed (no renderer was available or
  approved), so legibility at 100% is unverified.
- SUBMISSION_READY = false.
- Under the project rules, do not upload this source to an external web
  compile service. Build it on a machine where a local TeX installation is
  approved.

## Requirements

- A TeX distribution with `pdflatex` and `bibtex`, e.g. TeX Live 2023 or
  later. Required packages:
  - loaded by `paper/main.tex`: microtype, graphicx, subcaption, booktabs,
    hyperref, amsmath, amssymb, mathtools, amsthm, cleveref, pgfplots
    (compat 1.17);
  - required by `icml2026.sty`: times, fancyhdr, xcolor, algorithm,
    algorithmic, natbib, eso-pic, forloop, url, caption.
- `curl` or `wget`, `sha256sum` or `shasum`, and `unzip` or `python3`.
  These are used only by `fetch_style.sh`.

## Exact commands

From the directory that contains this file:

```sh
sh fetch_style.sh             # official ICML 2026 style -> paper/, SHA-256 verified
cd paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Equivalent, if latexmk is available: `cd paper && latexmk -pdf main.tex`.

The style is used unmodified in anonymous review mode
(`\usepackage{icml2026}`). Do not add `[accepted]` or `[preprint]`, and do
not change margins, fonts or spacing.

## Checks after building

```sh
grep -c "Warning.*undefined" main.log      # expect 0 (citations, references)
grep -n "Overfull" main.log | head          # inspect; fix any in the body
pdfinfo main.pdf | grep Pages
# page on which the Impact Statement starts (the 8-page limit applies before it):
n=$(pdfinfo main.pdf | awk '/^Pages/{print $2}')
for p in $(seq 1 "$n"); do pdftotext -f "$p" -l "$p" main.pdf - | grep -q "Impact Statement" && echo "Impact Statement on page $p"; done
```

- The body fits the ICML limit only if everything before the Impact
  Statement fits in 8 pages.
- The static estimate was about 5080 words before the appendix, including
  tables and captions. That number is not a page count.
- If the body is over 8 pages, move material to the appendix without
  changing any number. All numbers come from `paper/generated/numbers.tex`.
- The one red `[TODO: P5-REAL-02 ...]` marker is intentional. It marks
  evidence that has not been produced.

## Static check (no TeX needed)

```sh
python3 scripts/check_tex.py      # citations, refs, macros, environments, braces, anonymity patterns
```

## Regenerating tables and numbers

`scripts/make_paper_assets.py` reads the full v1 per-example rows
(`results/raw/p5_first_run_v1_test/*.jsonl`). This bundle does not include
them because of their size, so run it from the repository checkout at the
commit that contains this bundle. The generated files in `paper/generated/`
are already included, so building the PDF does not need regeneration.

## Tests and runs

Standard library (GMM controls and the test machinery):

```sh
python3 -m unittest discover -s tests         # about 13 s, 1 thread; the torch fixtures SKIP without torch
python3 scripts/run_syn_learn.py --config configs/p5_syn_learn_01.json --split test --out /tmp/p5_syn_rerun
```

Neural flow arms (round 6) need an isolated venv with a CPU-only torch; the
exact versions and the wheel hash are in `results/env/torch_env.json`:

```sh
python3 -m venv .venv-p5 && . .venv-p5/bin/activate
pip download --no-deps --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu torch==2.14.1+cpu -d wheels
sha256sum wheels/torch-*.whl                  # expect 5e38154c8896d426a5df58bf2276b603e82184dfae4f54d77dbb212f36e34f37
pip install --index-url https://pypi.org/simple wheels/torch-*.whl numpy==2.4.6
OMP_NUM_THREADS=1 python scripts/run_syn_learn_fm.py --smoke --out /tmp/p5_fm_smoke          # train-only timing smoke (seed 999)
OMP_NUM_THREADS=1 python scripts/run_syn_learn_fm.py --out /tmp/p5_fm_rerun --cpu-spent-before 0   # full re-run into a fresh root
```

The runner refuses to write into an existing output root. A re-run with the
same wheel should reproduce `results/raw/p5_syn_learn_01_fm_test/summary.json`
up to timings (seeds are fixed; CPU kernels of a different torch build may
differ in the last bits).

`run_syn_learn.py` refuses to overwrite an existing output directory.
Pre-registered runs must not be repeated for reporting. A re-run into a
fresh directory is only a reproducibility check. It should reproduce
`results/raw/p5_syn_learn_01_test/summary.json` up to timings, because the
seeds are fixed.
