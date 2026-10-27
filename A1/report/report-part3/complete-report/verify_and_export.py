"""Verify the compiled report, export reading copies, and package LaTeX source."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
from html.parser import HTMLParser
from html import escape, unescape
from urllib.parse import urlsplit, unquote

import pymupdf
import pypandoc
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QA = HERE / "qa"
OUT = ROOT / "output/pdf"
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

preservation = json.loads((HERE / "preservation.json").read_text())
for name, expected in preservation.items():
    assert sha(ROOT / name) == expected, f"Original file changed: {name}"
evidence = json.loads((QA / "evidence.json").read_text())
assert evidence["status"] == "passed"
log = (HERE / "report.log").read_text(errors="replace")
for bad in ["Overfull", "undefined", "Missing character", "multiply defined"]:
    assert bad not in log, f"Compiler warning: {bad}"
doc = pymupdf.open(HERE / "report.pdf")
fulltext = "\n".join(p.get_text() for p in doc)
assert "??" not in fulltext and "\ufffd" not in fulltext
assert not re.search(r"\b(TODO|TBD|PLACEHOLDER)\b", fulltext)
for phrase in ["91.94", "88.50", "88.48", "85.55", "84.26", "0.9189",
               "421,642", "532", "806", "CUDA", "Reproducibility and Evidence"]:
    assert phrase in fulltext, phrase
for model, cost in evidence["cuda_costs"].items():
    assert cost["training"]["device"] == cost["test"]["hardware"]["device"] == "cuda"
    assert f"{cost['training']['training_seconds']:.2f}" in fulltext
    assert f"{cost['test']['inference']['forward_batch_ms']:.3f}" in fulltext
    assert f"{cost['test']['inference']['evaluation_seconds']:.3f}" in fulltext
assert "MPS" not in fulltext
toc = doc.get_toc()
assert len(toc) > 30
for level, title, page in toc:
    assert 1 <= page <= len(doc)
    # The PDF text has typographic ligatures; normalise both for matching.
    normalize = lambda s: s.replace("\ufb01", "fi").replace("\ufb02", "fl")
    assert normalize(title).lower() in normalize(doc[page-1].get_text()).lower(), title
page_info = []
for i, page in enumerate(doc):
    assert len(page.get_text()) > 300, f"Nearly empty page {i+1}"
    for block in page.get_text("blocks"):
        x0, y0, x1, y1 = block[:4]
        assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width+1 and y1 <= page.rect.height+1
    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(QA / f"page-{i+1:02}.png")
    page_info.append({"page": i+1, "characters": len(page.get_text())})
for start in range(0, len(doc), 4):
    canvas = Image.new("RGB", (900, 1280), "#bbb")
    for j in range(min(4, len(doc)-start)):
        im = Image.open(QA / f"page-{start+j+1:02}.png")
        im.thumbnail((440, 615))
        canvas.paste(im, (j % 2*450+5, j//2*640+20))
        ImageDraw.Draw(canvas).text((j % 2*450+10, j//2*640+2),
                                    f"PDF page {start+j+1}", fill="black")
    canvas.save(QA / f"contact-{start//4+1:02}.png")
(QA / "report_text.txt").write_text(fulltext, encoding="utf-8")

# Make readable alternate formats from the canonical LaTeX, with resolved refs.
aux = (HERE / "report.aux").read_text()
labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))
bib = dict(re.findall(r"\\bibcite\{([^}]+)\}\{([^}]+)\}", aux))
def expand(text):
    def replace(match):
        path = HERE / match[1]
        if not path.suffix:
            path = path.with_suffix(".tex")
        return expand(path.read_text(encoding="utf-8"))
    return re.sub(r"\\(?:input|tabledata)\{([^}]+)\}", replace, text)

source = expand((HERE / "report.tex").read_text(encoding="utf-8"))
source = source[source.index(r"\section*{Abstract}"):source.rindex(r"\end{document}")]
source = re.sub(r"\\addcontentsline\{[^}]+\}\{[^}]+\}\{[^}]+\}", "", source)
source = re.sub(r"\\(?:tableofcontents|clearpage|phantomsection|begingroup|endgroup|appendix)\b", "", source)
source = re.sub(r"\\pagenumbering\{[^}]+\}", "", source)
source = re.sub(r"\\setlength\{[^}]+\}\{[^}]+\}", "", source)
source = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels[m[1]], source)
source = re.sub(r"\\cite\{([^}]+)\}",
                lambda m: "[" + ", ".join(bib[k] for k in m[1].split(",")) + "]", source)
defs = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]+)\}",
                      (HERE / "generated/derived_values.tex").read_text()))
for name, value in defs.items():
    source = re.sub(r"\\" + name + r"\b(?:\{\})?", value, source)
source = re.sub(r"\\includegraphics(?:\[[^]]+\])?\{([^}]+)\}",
                lambda m: r"\includegraphics{figures/" + m[1].replace(".pdf", ".png") + "}", source)
source = re.sub(r"\\begin\{thebibliography\}\{[^}]+\}",
                r"\\section*{References}\n\\begin{enumerate}", source)
source = source.replace(r"\end{thebibliography}", r"\end{enumerate}")
source = re.sub(r"\\bibitem\{[^}]+\}", r"\\item", source)
sec = 0
sub = 0
def number_heading(m):
    global sec, sub
    kind, star, name = m.groups()
    if star:
        return m[0]
    if kind == "section":
        sec += 1
        sub = 0
        number = str(sec) if sec <= 7 else chr(ord("A") + sec-8)
    else:
        sub += 1
        number = (str(sec) if sec <= 7 else chr(ord("A") + sec-8)) + "." + str(sub)
    return "\\" + kind + "{" + number + " " + name + "}"
source = re.sub(r"\\(section|subsection)(\*?)\{([^}]+)\}", number_heading, source)
counts = {"figure": 0, "table": 0}
def number_captions(match):
    kind, content = match.groups()
    counts[kind] += 1
    prefix = f"{kind.capitalize()} {counts[kind]}. "
    content = content.replace(r"\caption{", r"\caption{" + prefix, 1)
    return r"\begin{" + kind + "}" + content + r"\end{" + kind + "}"
source = re.sub(r"\\begin\{(figure|table)\}(.*?)\\end\{\1\}", number_captions, source, flags=re.S)
export_source = HERE / "generated/reading_copy.tex"
export_source.write_text(source, encoding="utf-8")
css = """body{font-family:Georgia,serif;line-height:1.6;max-width:1000px;margin:3rem auto;padding:0 2rem;color:#202428}
h1,h2,h3,h4{font-family:Arial,sans-serif;line-height:1.3;color:#183f42}
h1{margin-top:2.5rem}h2{margin-top:2rem}img{max-width:100%;height:auto}
figure{margin:2rem auto;text-align:center}figcaption{font-size:.92rem;color:#444}
table{border-collapse:collapse;margin:1.5rem auto;max-width:100%;font-size:.92rem}
th,td{padding:.45rem .6rem;border-bottom:1px solid #ddd}th{border-top:2px solid #333}
caption{margin-bottom:.7rem;text-align:left}pre{overflow:auto;background:#f3f5f5;padding:1rem}
math[display=block]{margin:1rem auto;overflow:auto}a{color:#195e55}
nav{padding:1rem 2rem;background:#f3f5f5}header{text-align:center}
@media print{nav{display:none}body{font-size:10pt;max-width:none}h1{break-before:page}figure,table{break-inside:avoid}}
"""
(HERE / "report.css").write_text(css)
common = ["--standalone", "--metadata=title:Fashion-MNIST Image Classification",
          "--metadata=subtitle:A Comparison of Five Neural Models",
          "--metadata=author:Team G-09 | CO3133 | October 2026",
          "--resource-path=" + str(HERE)]
pypandoc.convert_file(str(export_source), "html5", format="latex",
                      outputfile=str(HERE / "report.html"),
                      extra_args=common + ["--toc", "--toc-depth=2", "--mathml",
                                           "--css=" + str(HERE / "report.css"),
                                           "--embed-resources"])
pypandoc.convert_file(str(export_source), "gfm+tex_math_dollars", format="latex",
                      outputfile=str(HERE / "report.md"), extra_args=["--wrap=none"])

class HTMLCheck(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr"}
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.hrefs = []
        self.images = 0
        self.stack = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            assert a["id"] not in self.ids
            self.ids.add(a["id"])
        if tag == "a":
            self.hrefs.append(a.get("href", ""))
        if tag == "img":
            self.images += 1
            assert a.get("src", "").startswith("data:"), "HTML should be self-contained"
            assert a.get("alt", "").strip(), "Image needs descriptive alt text"
        if tag not in self.VOID:
            self.stack.append(tag)
    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        assert self.stack and self.stack[-1] == tag, (tag, self.stack[-4:])
        self.stack.pop()
def caption_text(figure):
    caption = re.search(r"<figcaption>(.*?)</figcaption>", figure, re.S)[1]
    return " ".join(unescape(re.sub(r"<[^>]+>", "", caption)).split())

html = (HERE / "report.html").read_text(encoding="utf-8")
html = re.sub(r"<figure\b.*?</figure>",
              lambda m: m[0].replace("<img ", '<img alt="' + escape(caption_text(m[0]), quote=True) + '" ', 1),
              html, flags=re.S)
(HERE / "report.html").write_text(html, encoding="utf-8")
parser = HTMLCheck()
parser.feed(html)
assert not parser.stack
for href in parser.hrefs:
    u = urlsplit(href)
    if href.startswith("#"):
        assert unquote(u.fragment) in parser.ids
    elif not u.scheme and u.path:
        assert (HERE / unquote(u.path)).exists()
assert parser.images == counts["figure"]
assert "<math" in html and counts["figure"] >= 10
md = (HERE / "report.md").read_text(encoding="utf-8")
def markdown_figure(match):
    figure = match[0]
    path = re.search(r'<img[^>]*src="([^"]+)"', figure)[1]
    caption = caption_text(figure).replace("[", r"\[").replace("]", r"\]")
    return f"![{caption}]({path})\n\n*{caption}*"
md = re.sub(r"<figure\b.*?</figure>", markdown_figure, md, flags=re.S)
(HERE / "report.md").write_text(md, encoding="utf-8")
for name in re.findall(r"!\[[^]]*\]\(([^)]+)\)", md):
    assert (HERE / name).exists()
for phrase in ["91.94", "88.50", "88.48", "85.55", "84.26"]:
    assert phrase in html and phrase in md

report = {"status": "passed", "pages": len(doc),
          "figures": counts["figure"], "tables": counts["table"],
          "bookmarks_checked": len(toc), "protected_files_unchanged": len(preservation),
          "ai_usage_sha256": sha(ROOT / "AI_USAGE.md"),
          "saved_predictions_checked": 50000, "compiler_warnings": [],
          "cost_models_on_cuda": list(evidence["cuda_costs"]),
          "baseline_cuda_predictions_verified": 20000,
          "checks": {"text_within_page_bounds": True, "all_pages_rendered": True,
                     "no_empty_pages": True, "no_unresolved_references": True,
                     "bookmark_destinations_checked": True,
                     "html_figures_embedded": parser.images,
                     "html_structure_and_links": True, "markdown_figures_exist": True},
          "page_details": page_info, "pdf_sha256": sha(HERE / "report.pdf"),
          "evidence_sha256": sha(QA / "evidence.json")}
(QA / "verification.json").write_text(json.dumps(report, indent=2)+"\n")
shutil.copy2(HERE / "report.pdf", OUT / "assignment1_complete_report.pdf")
shutil.copy2(HERE / "report.html", OUT / "assignment1_complete_report.html")
package = HERE / "assignment1_complete_source.zip"
with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
    files = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in
             {".tex", ".pdf", ".png", ".py", ".md", ".css", ".html", ".json"}]
    for p in files:
        if "qa" in p.parts and p.name not in {"evidence.json", "verification.json", "visual_review.md"}:
            continue
        if p.name == "report.pdf":
            continue
        z.write(p, p.relative_to(HERE))
shutil.copy2(package, OUT / package.name)
print(json.dumps({k: report[k] for k in ["status", "pages", "figures", "tables",
                                         "protected_files_unchanged"]}, indent=2))
