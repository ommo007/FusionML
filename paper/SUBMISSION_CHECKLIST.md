# Submission Checklist — arXiv preprint + MLSys 2027

Target plan (decided 2026-07-24):
- **arXiv preprint: submit mid-August 2026** (timestamps before PhD apps; both
  venues permit preprints)
- **MLSys 2027: submit at its deadline, expected ~Oct 30, 2026** (verify when
  the official CFP posts at mlsys.org — historically late October; conference
  May 2027). Chosen over ICLR 2027 (abstract Sep 19 / paper Sep 24, Brazil):
  systems-characterization content fits MLSys reviewers; ICLR risks "no
  learning contribution" rejections and forces a 5-week rush.
- Do NOT dual-submit the same paper to both. If MLSys rejects (decisions
  ~Feb 2027), resubmit to EuroMLSys / ATC / or ICLR 2028.

## Phase 0 — before arXiv (this week → mid-August)

- [ ] **Author pass on main.tex**: read every sentence; fix affiliation
      footnote (exact institute name); decide acknowledgments (benchmark
      contributors — ask friends how they want to be credited).
- [ ] **Verify every number** against `benchmarks/PAPER_READINESS.md` one
      final time (all draft numbers were taken from it on 2026-07-24).
- [ ] **Verify all 12 BibTeX entries** against DBLP/arXiv (titles/venues
      written from memory — check each; especially CoDL author list and
      Splitwise venue details).
- [ ] Optional strengtheners (not blockers):
      - [ ] M4 Pro full-depth re-run (previous run invalidated — swapping)
      - [ ] M3 Pro dynamic-arm re-run (two outlier cells)
      - [ ] Per-shape calibration in the mlx-lm patch (likely 1.25→1.3+ TTFT;
            update Fig. 3 + abstract if done)
- [ ] **Repo hygiene for public scrutiny**: README numbers match paper;
      PAPER_READINESS.md is consistent; tag the commit the paper describes
      (`git tag arxiv-v1 && git push --tags`).
- [ ] Get one external read: research mentor / the professor writing your
      letter. (Also the moment to ask about co-authorship/supervision.)

## Phase 1 — arXiv submission (mid-August)

1. Account: register at arxiv.org with om.mohite@vit.edu.in (academic domain
   helps auto-endorsement). If endorsement is still requested for cs.DC,
   ask your IGARSS supervisor or any arXiv-published colleague.
2. Categories: **cs.DC (primary)**, cross-list **cs.LG** and **cs.PF**.
3. License: arXiv non-exclusive license v1.0 (default) — do NOT pick CC-BY
   unless you've decided; MLSys copyright is compatible with the default.
4. Upload: `main.tex`, `references.bib` (or the generated `main.bbl` —
   arXiv prefers .bbl included), `figures/*.pdf`. Build locally first:
   `pdflatex && bibtex && pdflatex && pdflatex`.
5. Title/abstract into the form (plain text, no LaTeX macros — expand \x{}).
6. Submit → announcement next business day (Sun–Thu 20:00 ET cycles).
   **Record the arXiv ID** → add to CV, SoP, PhD applications, repo README.

## Phase 2 — MLSys 2027 submission (October)

1. **Watch mlsys.org for the official CFP** (typically posted late summer).
   Confirm: exact deadline, page limit (historically 10 pages + refs),
   template (`mlsys2027.sty`, ICML-derived two-column), review platform
   (CMT or OpenReview — link will be in CFP).
2. ~~Port to template~~ DONE — main.tex is already in official MLSys format
   (mlsys2026.sty, 5 pages of the 10 allowed, incl. architecture + timeline
   TikZ figures). When mlsys2027.sty posts: rename the \usepackage and .bst
   references (historically a year bump).

   **main.tex as committed is the arXiv variant** (real name, no venue
   notice, no public email — see the file's header comment). Before
   submitting to MLSys, make a copy and apply the swap described there:
   - Blind submission: drop `[accepted]`, drop the
     `\renewcommand{\printAffiliationsAndNotice}` override (restores the
     stock "Preliminary work. Under review..." notice + author
     anonymization).
   - Camera-ready (if accepted): keep `[accepted]`, drop the same override,
     add back `\mlsyscorrespondingauthor{Om Mohite}{<email>}`.
3. **Anonymize (double-blind)**:
   - Remove author block.
   - Replace github.com/ommo007/FusionML with an **anonymous mirror**
     (anonymous.4open.science) — grep the PDF for "ommo007", "Mohite",
     "Vishwakarma", "FusionML" (consider renaming the system in submission
     if the repo is easily searchable — check CFP anonymity policy level).
   - The arXiv preprint may exist (MLSys follows standard prior-preprint
     policy) but do not cite it from the submission in a de-anonymizing way.
4. Artifact evaluation: MLSys runs AE post-acceptance — the repo already
   qualifies (one-command suite, environment-stamped JSONs, transcripts).
   Note intent to participate in the submission form if asked.
5. Register abstract by any abstract-deadline the CFP specifies.
6. Submit PDF; confirm receipt email; calendar the rebuttal window.

## Phase 3 — after submission

- [ ] CV/SoP line: "Under review, MLSys 2027" + arXiv ID.
- [ ] PhD applications (Dec 1–15) cite the arXiv version.
- [ ] Rebuttal (likely Dec–Jan): PAPER_READINESS.md is the rebuttal
      ammunition file — keep it current if any new runs land.
- [ ] Decisions ~Feb 2027. Accept → camera-ready + AE. Reject → triage
      reviews, resubmit EuroMLSys (spring) or ATC '27.

## Known weaknesses to expect in review (prepared answers)

1. "Single-vendor platform" → mechanism generalizes to lazy-graph frameworks
   with device streams; Apple Silicon is the dominant deployed unified-memory
   platform; CoDL precedent for platform-specific characterization.
2. "Modest speedups" → free at API level, no accuracy loss (token-identical),
   scoped claims with mechanism; plus the boundary characterization is the
   contribution, not just the wins.
3. "Synthetic blocks" → full-depth control + real-checkpoint end-to-end with
   transcripts (Sections 5.3, 5.4) — this objection is pre-answered.
4. "Why not quantized models?" → limitation; split is orthogonal to weight
   dtype for the activation-row dimension; future work.
