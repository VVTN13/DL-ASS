# Stage-1 PDF visual review

The duplicate-audit update was checked in its standalone PNG and on rendered
PDF page 8. All 15 pages were rendered and passed the automated checks. The
current PDF SHA-256 is
`b53b8c89b395ae2c3a1b89b4879ab832647d627c6665be0053d6f104f519ba55`.
The other visual observations below are retained from the earlier full review.

- Cover, Vietnamese names, logo, Roman front-matter numbering, and Arabic body
  numbering render correctly.
- All nine requested subsections appear, with nine figures and four tables.
- The former duplicate-audit table is now Figure 2, with separate panels for
  within-split duplicate excess and shared hashes between splits. All six zero
  counts are visibly labeled; the values come directly from the computed audit.
- Chart labels, table columns, equations, captions, and references are readable.
- No clipped content, overlapping body text, empty pages, or isolated spillover
  page remains. The References section starts on its own final page.
- The table of contents and PDF bookmarks resolve to the corresponding pages.

Automated evidence, hashes, bounds checks, caption counts, and rendering details
are in `verification.json`. The EDA's numerical checks and CUDA environment are
in `../data/eda_metrics.json`. Rebuild and review again after editing the report.
