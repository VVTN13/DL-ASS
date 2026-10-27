"""Continue the supplied template PDF without rebuilding its existing pages.

From A1/report, compile template_continuation.tex, then run this script with
../.venv/Scripts/python.exe. The original PDF is read only. The merged output
keeps its cover and Sections 1--2.3, corrects three figure references, and
replaces the empty 2.4 heading with the shared editable LaTeX continuation.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE / "sections" / "Report_Style_Template (7).pdf"
OUTPUT = ROOT / "output" / "pdf" / "assignment1_template_continued.pdf"
QA = HERE / "qa" / "template_continued"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_font(doc, page_number, suffix):
    entry = next(f for f in doc[page_number].get_fonts() if f[3].endswith(suffix))
    return doc.extract_font(entry[0])[3]


def isolated_region(doc, page_number, rect):
    """Remove content outside the region, including otherwise hidden PDF text."""
    isolated = fitz.open()
    isolated.insert_pdf(doc, from_page=page_number, to_page=page_number)
    page = isolated[0]
    for outside in [fitz.Rect(0, 0, page.rect.width, rect.y0),
                    fitz.Rect(0, rect.y1, page.rect.width, page.rect.height)]:
        if not outside.is_empty:
            page.add_redact_annot(outside, fill=(1, 1, 1))
    page.apply_redactions(images=2, graphics=1)
    return isolated


def replace_reference(page, old, new, font_bytes):
    matches = page.search_for(old)
    assert len(matches) == 1, (old, matches)
    rect = matches[0]
    spans = [s for b in page.get_text("dict")["blocks"] if b["type"] == 0
             for line in b["lines"] for s in line["spans"]
             if fitz.Rect(s["bbox"]).intersects(rect)]
    origin_y = spans[0]["origin"][1]
    page.add_redact_annot(rect, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    page.insert_text((rect.x0, origin_y), new, fontsize=9.9626,
                     fontname="tiro")


def main():
    before = {"template": sha(SOURCE), "AI_USAGE.md": sha(ROOT / "AI_USAGE.md")}
    original = fitz.open(SOURCE)
    continuation = fitz.open(HERE / "template_continuation.pdf")
    # The retained template header occupies y=0..102. No body material may
    # fall under the header when the branding is applied to appended pages.
    for page in continuation:
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            text = " ".join(s["text"] for line in block["lines"] for s in line["spans"])
            if "HO CHI MINH CITY" not in text and block["bbox"][1] < 790:
                assert block["bbox"][1] > 102, (block["bbox"], text)
    result = fitz.open()
    result.insert_pdf(original, from_page=0, to_page=8)
    result.insert_pdf(continuation)
    total = len(result) - 1  # the cover is unnumbered
    body_font = original_font(original, 1, "CMR10")
    footer_font = original_font(original, 1, "CMTT8")
    header_region = fitz.Rect(0, 0, original[9].rect.width, 102)
    footer_region = fitz.Rect(0, 790, original[9].rect.width, original[9].rect.height)
    figure_region = fitz.Rect(0, 102, original[9].rect.width, 480)
    header_source = isolated_region(original, 9, header_region)
    footer_source = isolated_region(original, 9, footer_region)
    figure_source = isolated_region(original, 9, figure_region)

    # Reuse the actual supplied header and footer so every appended page
    # carries the same institutional branding and page geometry.
    for index in range(9, len(result)):
        page = result[index]
        header = fitz.Rect(0, 0, page.rect.width, 102)
        page.draw_rect(header, fill=(1, 1, 1), color=None, overlay=True)
        page.show_pdf_page(header, header_source, 0, clip=header)
        footer = fitz.Rect(0, 790, page.rect.width, page.rect.height)
        page.add_redact_annot(footer, fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        page.show_pdf_page(footer, footer_source, 0, clip=footer)

    # Restore the original class-similarity figure above the new 2.4 text.
    page = result[9]
    figure_region = fitz.Rect(0, 102, page.rect.width, 480)
    page.show_pdf_page(figure_region, figure_source, 0, clip=figure_region)
    assert len(re.findall(r"2\.4\s+Preprocessing", page.get_text())) == 1

    # Correct the existing EDA's off-by-one references to Figures 2, 3, 4.
    for page_index, old, new in [(4, "Figure 3", "Figure 2"),
                                  (5, "Figure 4", "Figure 3"),
                                  (5, "Figure 5", "Figure 4")]:
        replace_reference(result[page_index], old, new, body_font)

    # Update the printed total throughout the preserved and appended pages.
    for index in range(1, len(result)):
        page = result[index]
        matches = page.search_for("Page ")
        assert len(matches) == 1, (index, matches)
        page.add_redact_annot(fitz.Rect(487, 806, 541, 821), fill=(1, 1, 1))
        page.apply_redactions(images=0, graphics=0)
        label = f"Page {index}/{total}"
        x = 538.58 - fitz.get_text_length(label, fontname="cour", fontsize=6.97385)
        page.insert_text((x, 815.26), label, fontname="cour", fontsize=6.97385)

    # Add 2.5 to the existing table of contents and preserve its established
    # subsection numbering. The original 2.4 page remains page 9.
    findings_page = next(i for i, p in enumerate(result)
                         if i > 8 and "Key Findings and Implications" in p.get_text())
    toc_page = result[1]
    toc_page.insert_font(fontname="OriginalBody", fontbuffer=body_font)
    # The template embeds a subset; use a complete Times font for the new
    # line's title rather than risk a missing character in that subset.
    toc_page.insert_text((71.637, 307.96), "2.5", fontname="OriginalBody", fontsize=9.9626)
    toc_page.insert_text((94.551, 307.96), "Key Findings and Implications for Methodology",
                         fontname="tiro", fontsize=10)
    toc_page.insert_text((390, 307.96), ". . . . . . . . . . . . . . . . . . . . . .",
                         fontname="tiro", fontsize=9.9626)
    number = str(findings_page)
    font = fitz.Font(fontbuffer=body_font)
    toc_page.insert_text((538.58-font.text_length(number, fontsize=9.9626), 307.96),
                         number, fontname="OriginalBody", fontsize=9.9626)

    toc = [[1, "Introduction", 3],
           [2, "Background and Motivation", 3], [2, "Problem Statement", 3],
           [2, "Objectives and Scope", 3], [2, "Proposed Approach", 3],
           [1, "Dataset and Exploratory Analysis", 4],
           [2, "Dataset Description", 4], [2, "Data Preprocessing", 4],
           [2, "Exploratory Data Analysis and Preprocessing", 5],
           [2, "Preprocessing", 10]]
    for i, p in enumerate(result):
        if i < 9:
            continue
        lines = p.get_text().splitlines()
        for j, line in enumerate(lines[:-1]):
            if re.fullmatch(r"2\.4\.[1-7]", line):
                toc.append([3, line + " " + lines[j+1], i+1])
    toc.append([2, "Key Findings and Implications for Methodology", findings_page+1])
    result.set_toc(toc)
    result.set_metadata({"title": "Assignment 1 - Dataset, EDA and Preprocessing",
                         "author": "Team G-09", "subject": "Completed Section 2"})
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.save(OUTPUT, garbage=4, deflate=True)
    result.close()

    # Verify the finished artifact, not just the source or TeX build status.
    QA.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(OUTPUT)
    text = "\n".join(p.get_text() for p in pdf)
    assert "??" not in text and "\ufffd" not in text
    assert "Page 9/" + str(total) in pdf[9].get_text()
    assert "Page " + str(total) + "/" + str(total) in pdf[-1].get_text()
    assert re.search(r"Figure 8:", text)
    assert re.search(r"Table 5:", text) and re.search(r"Table 6:", text)
    assert "0.286209" in text and "0.353160" in text
    assert "2.4.6" in text and "Key Findings and Implications for Methodology" in text
    for log_name in ["template_continuation.log", "report.log"]:
        log = (HERE / log_name).read_text(encoding="utf-8", errors="replace")
        assert "Overfull" not in log, log_name
        assert "Missing character" not in log, log_name
        assert "undefined references" not in log.lower(), log_name
    for index, page in enumerate(pdf):
        assert len(page.get_text().strip()) > 50
        for block in page.get_text("dict")["blocks"]:
            if block["type"] == 0:
                assert page.rect.contains(fitz.Rect(block["bbox"])), (index+1, block["bbox"])
        page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25)).save(QA / f"page-{index+1:02d}.png")
    # Review sheets cover every page, including the retained source pages.
    for start in range(0, len(pdf), 6):
        sheet = Image.new("RGB", (1270, 1810), "#e0e5e9")
        draw = ImageDraw.Draw(sheet)
        for j, index in enumerate(range(start, min(start+6, len(pdf)))):
            im = Image.open(QA / f"page-{index+1:02d}.png").convert("RGB")
            im = ImageOps.contain(im, (610, 550))
            x = (j % 2)*630 + 10 + (610-im.width)//2
            y = (j // 2)*600 + 30
            sheet.paste(im, (x, y))
            draw.text(((j%2)*630+15, y-20), f"PDF page {index+1}", fill="#243746")
        sheet.save(QA / f"contact-{start//6+1:02d}.png")
    assert before["template"] == sha(SOURCE)
    assert before["AI_USAGE.md"] == sha(ROOT / "AI_USAGE.md")
    verification = {"passed": True, "output": str(OUTPUT), "pages": len(pdf),
                    "output_sha256": sha(OUTPUT), "original_template_sha256": before["template"],
                    "original_template_unchanged": True, "AI_USAGE_unchanged": True,
                    "figure_references_corrected": 3, "all_pages_rendered": True,
                    "no_overfull_boxes_or_missing_glyphs": True,
                    "new_sections": ["2.4 Preprocessing", "2.5 Key Findings and Implications for Methodology"]}
    (QA / "verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    # Export a paste-ready fragment for the user's original LaTeX project.
    # Replace report-specific macros and expand the custom figure helper.
    fragment = (HERE / "sections/024_preprocessing.tex").read_text(encoding="utf-8")
    fragment += "\n" + (HERE / "sections/025_key_findings.tex").read_text(encoding="utf-8")
    for macro, value in [(r"\TrainMean", "0.286209"), (r"\TrainStd", "0.353160"),
                         (r"\ZeroPercent", "50.21"), (r"\MostSimilarCosine", "0.9873")]:
        fragment = fragment.replace(macro, value)
    helper = r"\reportfigure{07_preprocessing_representations}{1.0}{"
    start = fragment.index(helper)
    end = fragment.index("}{fig:representations}", start)
    caption = fragment[start+len(helper):end]
    figure = ("\\begin{figure}[htbp]\\centering\n"
              "\\includegraphics[width=\\linewidth]{preprocessing_representations.pdf}\n"
              "\\caption{"+caption+"}\\label{fig:representations}\n\\end{figure}")
    fragment = fragment[:start] + figure + fragment[end+len("}{fig:representations}"):]
    introduction = ("% Replace the existing empty 2.4 heading with this entire fragment.\n"
                    "% Paste after Figure 7; 2.4 and 2.5 numbering continues automatically.\n"
                    "% Required packages: amsmath, amssymb, graphicx, booktabs, tabularx, float, placeins.\n"
                    "% Upload preprocessing_representations.pdf beside the main .tex file.\n\n")
    export_tex = OUTPUT.parent / "section2_continuation.tex"
    export_tex.write_text(introduction+fragment, encoding="utf-8")
    archive = OUTPUT.parent / "section2_continuation_source.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(export_tex, "section2_continuation.tex")
        bundle.write(HERE / "figures/07_preprocessing_representations.pdf", "preprocessing_representations.pdf")
        bundle.writestr("README.txt", "Replace the empty 2.4 heading in your original LaTeX project with section2_continuation.tex.\n"
                        "Place preprocessing_representations.pdf next to the project's main .tex file.\n"
                        "Use the packages listed in the fragment. The fragment includes Sections 2.4 and 2.5.\n"
                        "Correct existing EDA references: Class Distribution -> Figure 2; Individual Image Variation -> Figure 3; "
                        "Pixel intensity and background -> Figure 4.\n"
                        "The continued PDF preserves the supplied cover and earlier sections; model training was not rerun.\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
