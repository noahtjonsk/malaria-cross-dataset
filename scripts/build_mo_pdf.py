"""Render a Markdown document to an A4 PDF with headless Edge.

    python scripts/build_mo_pdf.py docs/methodology_overview_v3.md
    python scripts/build_mo_pdf.py docs/elsa_report.md --out some/other.pdf

Images are inlined, so the PDF does not depend on relative paths; links to
repository files are pointed at GitHub. Output: same name, .pdf.
"""
import argparse
import base64
import mimetypes
import re
import subprocess
import tempfile
from pathlib import Path

import markdown
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/noahtjonsk/malaria-cross-dataset/blob/main/"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """@page { size: A4; margin: 18mm 16mm; }
body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; margin-bottom: 4pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; margin-top: 18pt;
     page-break-after: avoid; }
h3 { font-size: 12pt; page-break-after: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 6pt 0; }
th, td { border: 1px solid #bbb; padding: 3pt 5pt; vertical-align: top; overflow-wrap: break-word; }
td code { overflow-wrap: anywhere; }
th { background: #eee; text-align: left; }
tr, img, pre { page-break-inside: avoid; }
img { max-width: 100%; display: block; margin: 6pt auto; }
code { font-size: 8.5pt; background: #f3f3f3; padding: 0 2px; }
pre { font-size: 7.5pt; line-height: 1.25; white-space: pre-wrap; background: #f6f6f6; padding: 6pt; }
pre code { background: none; padding: 0; }
blockquote { color: #444; border-left: 3px solid #ccc; margin-left: 0; padding-left: 10pt; }
a { color: #1a5fb4; text-decoration: none; }"""


def inline_images(html: str, base: Path) -> str:
    """Replace every local image path with a data URI."""
    def repl(m):
        f = (base / m.group(1)).resolve()
        if not f.exists():
            raise SystemExit(f"missing image {f}")
        mime = mimetypes.guess_type(f.name)[0]
        return f'src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}"'
    return re.sub(r'src="(?!data:|https?:)([^"]+)"', repl, html)


def absolute_links(html: str, base: Path) -> str:
    """Point links to repository files at GitHub, so they work outside the repo."""
    def repl(m):
        target = (base / m.group(1)).resolve()
        return f'href="{REPO}{target.relative_to(ROOT).as_posix()}{m.group(2) or ""}"'
    return re.sub(r'href="(?!https?:|#|mailto:)([^"#]+)(#[^"]*)?"', repl, html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--edge", default=EDGE)
    a = ap.parse_args()
    src = a.source.resolve()
    out = (a.out or src.with_suffix(".pdf")).resolve()
    body = markdown.markdown(src.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    body = absolute_links(inline_images(body, src.parent), src.parent)
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "doc.html"
        page.write_text(f"<!doctype html><html lang='en'><meta charset='utf-8'>"
                        f"<title>{src.stem}</title><style>{CSS}</style><body>{body}</body></html>",
                        encoding="utf-8")
        subprocess.run([a.edge, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={Path(tmp) / 'edge'}", f"--print-to-pdf={out}",
                        page.as_uri()], check=True, timeout=180,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not out.exists():
        raise SystemExit(f"Edge wrote no PDF at {out}")
    n = len(PdfReader(out).pages)
    print(f"wrote {out} ({n} pages)")


if __name__ == "__main__":
    main()
