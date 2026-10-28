"""Validate stage-1 report evidence and render every PDF page for visual QA.

From A1: .venv/Scripts/python.exe report/verify_report.py
Requires PyMuPDF==1.26.5 for PDF inspection; Pillow for contact sheets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    qa = HERE / "qa"
    qa.mkdir(exist_ok=True)
    evidence = json.loads((HERE / "data" / "eda_metrics.json").read_text())
    assert all(evidence["checks"].values())
    assert evidence["environment"]["device"] == "cuda"
    assert evidence["code_sha256"] == sha(HERE / "generate_eda.py")
    for relative, expected in evidence["generated_figures_sha256"].items():
        assert sha(HERE / relative) == expected, relative
    for relative, expected in evidence["protected_files_sha256"].items():
        assert sha(HERE.parent.parent / relative) == expected, relative
    with (HERE / "data" / "split_counts.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    for name, expected in (("train", 54000), ("validation", 6000), ("test", 10000)):
        assert sum(int(r[name]) for r in rows) == expected

    pdf = fitz.open(HERE / "report.pdf")
    texts, page_info, previews = [], [], []
    for number, page in enumerate(pdf, 1):
        text = page.get_text()
        texts.append(text)
        assert len(text.strip()) > 30, f"Empty page {number}"
        assert "??" not in text and "\ufffd" not in text, f"Unresolved text on page {number}"
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            rect = fitz.Rect(block["bbox"])
            assert rect.x0 >= -1 and rect.y0 >= -1, (number, rect)
            assert rect.x1 <= page.rect.width + 1 and rect.y1 <= page.rect.height + 1, (number, rect)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.2, 1.2), alpha=False)
        output = qa / f"page-{number:02d}.png"
        pixmap.save(output)
        previews.append(output)
        page_info.append({"page": number, "text_characters": len(text),
                          "width_pt": page.rect.width, "height_pt": page.rect.height,
                          "render": str(output.relative_to(HERE))})
    all_text = "\n".join(texts)
    compact = re.sub(r"\s+", " ", all_text)
    headings = ["Background and Motivation", "Problem Statement", "Objectives and Scope",
                "Proposed Approach", "Dataset Description", "Data Preparation and Splitting",
                "Exploratory Data Analysis", "Preprocessing", "Key Findings and Implications for Methodology"]
    for heading in headings:
        assert heading in compact, heading
    for name in ("Lê Phước Vũ", "Nguyễn Vũ Long", "Nguyễn Hoàng Quốc"):
        assert name in compact, name
    figures = [int(n) for n in re.findall(r"^Figure\s+(\d+)\.", all_text, re.MULTILINE)]
    tables = [int(n) for n in re.findall(r"^Table\s+(\d+)\.", all_text, re.MULTILINE)]
    assert sorted(figures) == list(range(1, 10)), figures
    assert sorted(tables) == list(range(1, 6)), tables
    assert "0.286209" in compact and "0.353160" in compact
    assert "50.21%" in compact
    log = (HERE / "report.log").read_text(encoding="utf-8", errors="replace")
    assert "Overfull" not in log, "Overfull box in LaTeX log"
    assert "undefined references" not in log.lower()
    assert "undefined citations" not in log.lower()
    assert "Missing character" not in log
    for level, title, target_page in pdf.get_toc():
        assert title in re.sub(r"\s+", " ", texts[target_page-1]), (title, target_page)

    # Contact sheets are a review aid; individual page renders retain detail.
    for offset in range(0, len(previews), 6):
        group = previews[offset:offset+6]
        sheet = Image.new("RGB", (1050, 1530), "#dce3e8")
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(group):
            page_image = Image.open(path).convert("RGB")
            page_image = ImageOps.contain(page_image, (500, 465))
            x = 15 + (i % 2) * 520 + (500 - page_image.width) // 2
            y = 30 + (i // 2) * 500
            sheet.paste(page_image, (x, y))
            draw.text((15+(i%2)*520, y-18), f"PDF page {offset+i+1}", fill="#253441")
        sheet.save(qa / f"contact-{offset//6+1:02d}.png")
    # Remove only obsolete renders produced by earlier builds of this script.
    for pattern, maximum in ((r"page-(\d+)\.png", len(pdf)),
                             (r"contact-(\d+)\.png", (len(pdf)+5)//6)):
        for path in qa.iterdir():
            match = re.fullmatch(pattern, path.name)
            if match and int(match[1]) > maximum:
                assert path.resolve().parent == qa.resolve()
                path.unlink()
    result = {"passed": True, "pdf_sha256": sha(HERE / "report.pdf"),
              "pages": len(pdf), "figure_captions": figures, "table_captions": tables,
              "required_subsections_present": headings,
              "checks": {"eda_checks_passed": True, "cuda_execution_recorded": True,
                         "figure_hashes_match": True, "protected_files_unchanged": True,
                         "section_headings_present": True, "vietnamese_names_extract": True,
                         "bookmarks_target_correct_pages": True,
                         "no_unresolved_references_or_missing_glyphs": True,
                         "no_overfull_latex_boxes": True, "text_within_page_bounds": True,
                         "all_pages_rendered": True},
              "page_details": page_info}
    (qa / "verification.json").write_text(json.dumps(result, indent=2)+"\n")
    (qa / "report_text.txt").write_text(all_text, encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "page_details"}, indent=2))


if __name__ == "__main__":
    main()
