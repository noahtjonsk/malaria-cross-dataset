"""The Methodology Overview as LaTeX, in the literature review's classicthesis style.

    python scripts/build_mo_latex.py              # write docs/methodology/mo_v3.tex, compile, copy the PDF
    python scripts/build_mo_latex.py --no-pdf     # only write the .tex

docs/methodology_overview_v3.md stays the source of truth; this script converts the
Markdown subset that document uses (headings, paragraphs, lists, pipe tables, images,
captions, code blocks, links, citations) and stops on anything it cannot convert, so the
LaTeX version never silently drops text. Compiled with TinyTeX:
pdflatex -> biber -> pdflatex -> pdflatex. The PDF is copied to
docs/methodology_overview_v3.pdf.
"""
import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs" / "methodology_overview_v3.md"
OUT_DIR = ROOT / "docs" / "methodology"
TEX = OUT_DIR / "mo_v3.tex"
PDF = ROOT / "docs" / "methodology_overview_v3.pdf"
REPO = "https://github.com/noahtjonsk/malaria-cross-dataset"
TEXBIN = Path(os.environ.get("APPDATA", "")) / "TinyTeX" / "bin" / "windows"

# Author-year as written in the Markdown -> key in docs/methodology/refs.bib
CITE_KEYS = {
    ("Rajaraman", "2018"): "rajaraman2018", ("Rajaraman", "2019"): "rajaraman2019",
    ("Ljosa", "2012"): "ljosa2012", ("Hung", "2017"): "hung2017",
    ("Loddo", "2019"): "loddo2019", ("Loddo", "2022"): "loddo2022",
    ("Kassim", "2021"): "kassim2021", ("Hou", "2026"): "hou2026", ("Zhao", "2020"): "zhao2020",
    ("Simonyan", "2015"): "simonyan2015", ("He", "2016"): "he2016", ("Sandler", "2018"): "sandler2018",
    ("Russakovsky", "2015"): "russakovsky2015", ("Kingma", "2015"): "kingma2015",
    ("Kapoor", "2023"): "kapoor2023", ("Beaulieu-Jones", "2017"): "beaulieujones2017",
    ("Di Cosmo", "2026"): "dicosmo2026", ("Reinhard", "2001"): "reinhard2001",
    ("Macenko", "2009"): "macenko2009", ("Xu", "2025"): "xu2025", ("Otsu", "1979"): "otsu1979",
    ("Qin", "2020"): "qin2020",
}
NOCITE = ["aiact2024", "mdr2017"]   # named in the text by regulation number, listed in References
NAME = r"(?:Di Cosmo|[A-Z][A-Za-z\-]+)"
AUTHORS = NAME + r"(?: et al\.| (?:&|and) [A-Z][A-Za-z\-]+)?"
ONE_CITE = rf"({AUTHORS}), (\d{{4}})"

LINEWIDTH, TABCOLSEP = 360.0, 6.0                                   # pt, classicthesis type area
CHAR_PT = {r"\small": 4.95, r"\footnotesize": 4.45, r"\scriptsize": 3.9}   # mean Palatino character width
UNICODE = {"§": r"\S{}", "−": r"\textminus{}", "–": "--", "×": r"$\times$", "²": r"\textsuperscript{2}",
           "µ": r"\textmu{}", "°": r"\textdegree{}", "…": r"\dots{}", "⁻⁴": r"\textsuperscript{\textminus{}4}"}
SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
TEXT_SPECIAL = dict(SPECIAL, _=r"\_\allowbreak{}")   # site_a, cell_images may break after _

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[american]{babel}
\usepackage{textcomp}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{longtable}
\usepackage{float}
\usepackage{fancyvrb}
\usepackage{ragged2e}
\usepackage[nochapters,beramono,eulermath,pdfspacing,dottedtoc]{classicthesis}

% type area (matches the DSS classicthesis template and the literature review)
\areaset[current]{360pt}{632pt}
\setlength{\footskip}{21pt}
\addtolength{\voffset}{48.4pt}
\setlength{\marginparwidth}{7em}
\setlength{\marginparsep}{2em}

\newcommand{\myTitle}{@@title@@}
\newcommand{\myName}{Noah Tjon Sien Kie}
\newcommand{\myStudentNumber}{@@student@@}
\newcommand{\myCommittee}{@@supervisor@@}
\newcommand{\myDate}{@@date@@}
\newcommand{\myStage}{Deliverable D3, Capstone Portfolio}
\newcommand{\repourl}{@@repo@@}

\usepackage{scrextend}
\newlength{\titlelogoskip}\setlength{\titlelogoskip}{-49.6pt}
\newlength{\titlegapA}\setlength{\titlegapA}{141.8pt}
\newlength{\titlegapB}\setlength{\titlegapB}{65pt}
\newlength{\titlegapC}\setlength{\titlegapC}{33pt}
\newlength{\bodytitleskip}\setlength{\bodytitleskip}{23.1pt}
\newlength{\bodygapA}\setlength{\bodygapA}{31.4pt}
\newlength{\bodygapB}\setlength{\bodygapB}{13.6pt}
\newlength{\backindent}\setlength{\backindent}{24pt}
\newlength{\backtopskip}\setlength{\backtopskip}{-15.7pt}

\definecolor{ctcite}{rgb}{0.05,0.05,0.50}
\hypersetup{colorlinks=true, linkcolor=ctcite, citecolor=ctcite, urlcolor=ctcite,
  pdftitle={\myTitle: Methodology Overview}, pdfauthor={\myName}, breaklinks=true}

\usepackage[style=apa,backend=biber,natbib=true]{biblatex}
\DeclareLanguageMapping{american}{american-apa}
\addbibresource{refs.bib}
\defbibheading{ctbib}[References]{%
  \section*{\spacedlowsmallcaps{#1}}%
  \markright{\spacedlowsmallcaps{#1}}}

\setlength{\LTcapwidth}{\linewidth}
\setlength{\emergencystretch}{3em}
\setlength{\LTpre}{6pt}
\setlength{\LTpost}{6pt}
"""

FRONT = r"""
\begin{document}
\frenchspacing
\raggedbottom
\pagestyle{empty}

\begin{titlepage}
  \begin{center}
    \vspace*{\titlelogoskip}
    \includegraphics[width=167pt]{tilburg-logo}
    \vspace{\titlegapA}

    {\huge\rmfamily\normalfont\spacedallcaps{\myTitle}\par}
    \vspace{\titlegapB}
    {\normalsize\spacedlowsmallcaps{\myName}\par}
    \vspace{\titlegapC}
    {\footnotesize\spacedlowsmallcaps{Methodology Overview}\par}
    \vspace{4pt}
    {\footnotesize\spacedlowsmallcaps{Data Science \& Society}\par}
    \vfill
  \end{center}
\end{titlepage}

\thispagestyle{empty}
\begin{addmargin}[\backindent]{0pt}
\noindent
\vspace*{\backtopskip}

\noindent\spacedlowsmallcaps{Student Number}

\vspace{7.5pt}\noindent\myStudentNumber

\vspace{15.1pt}

\noindent\spacedlowsmallcaps{Committee}

\vspace{7.5pt}\noindent\myCommittee

\vspace{15.1pt}

\noindent\spacedlowsmallcaps{Location}

\vspace{7.5pt}\noindent Tilburg University\\
School of Humanities and Digital Sciences\\
Department of Cognitive Science \&\\
Artificial Intelligence\\
Tilburg, The Netherlands

\vspace{15.1pt}

\noindent\spacedlowsmallcaps{Date}

\vspace{7.5pt}\noindent\myDate\ (\myStage)

\vspace{15.1pt}

\noindent\spacedlowsmallcaps{Repository}

\vspace{7.5pt}\noindent\url{\repourl}
\end{addmargin}

\clearpage
\pagestyle{scrheadings}
\rofoot[{\small\thepage}]{}
\setcounter{page}{1}
\thispagestyle{plain}

\begin{center}
  \vspace*{\bodytitleskip}
  {\LARGE\rmfamily\normalfont\spacedallcaps{Methodology Overview}\par}
  \vspace{\bodygapA}
  {\normalsize\spacedlowsmallcaps{\myName}\par}
  \vspace{\bodygapB}
\end{center}
"""


class Converter:
    def __init__(self, src: Path):
        self.src = src
        self.lines = src.read_text(encoding="utf-8").splitlines()
        self.sec = [0, 0]           # section, subsection
        self.appendix = False
        self.tables = 0             # captioned tables in the current numbering
        self.figures = 0
        self.cited = set()

    # ---------- inline ----------
    def code(self, s: str) -> str:
        out = []
        for ch in s:
            if ch in SPECIAL:
                out.append(SPECIAL[ch])
            else:
                out.append(ch)
            if ch in "/._":
                out.append(r"\allowbreak{}")
        return r"\texttt{" + "".join(out) + "}"

    def link_url(self, target: str) -> str:
        if target.startswith(("http://", "https://")):
            return target
        path = (self.src.parent / target).resolve().relative_to(ROOT).as_posix()
        return f"{REPO}/blob/main/{path}"

    def cite(self, authors: str, year: str) -> str:
        name = "Di Cosmo" if authors.startswith("Di Cosmo") else re.match(r"[A-Za-z\-]+", authors).group(0)
        key = CITE_KEYS.get((name, year))
        if key is None:
            raise SystemExit(f"no bib key for citation {authors}, {year}")
        self.cited.add(key)
        return key

    def inline(self, s: str) -> str:
        store = []

        def keep(latex: str) -> str:
            store.append(latex)
            return f"\x00{len(store) - 1}\x00"

        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                   lambda m: keep(r"\href{" + self.link_url(m.group(2)).replace("%", r"\%").replace("#", r"\#")
                                  + "}{" + self.inline(m.group(1)) + "}"), s)
        s = re.sub(r"`([^`]+)`", lambda m: keep(self.code(m.group(1))), s)
        # parenthetical citations: (A et al., 2018; B & C, 2017)
        s = re.sub(rf"\(({ONE_CITE}(?:; {ONE_CITE})*)\)",
                   lambda m: keep(r"\citep{" + ",".join(self.cite(a, y) for a, y in
                                                         re.findall(ONE_CITE, m.group(1))) + "}"), s)
        # textual citations: Hou et al. (2026), Kapoor and Narayanan (2023)
        s = re.sub(rf"({AUTHORS}) \((\d{{4}})\)",
                   lambda m: keep(r"\citet{" + self.cite(m.group(1), m.group(2)) + "}"), s)
        if "et al." in s:
            raise SystemExit(f"unconverted citation in: {s[:120]}")
        s = s.replace("⁻⁴", "\x01")
        s = "".join(TEXT_SPECIAL.get(ch, ch) for ch in s)
        s = s.replace("\x01", UNICODE["⁻⁴"])
        s = re.sub(r'"([^"]+)"', r"``\1''", s)
        s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)
        for ch, rep in UNICODE.items():
            s = s.replace(ch, rep)
        s = s.replace("MobileNetV2", r"\mbox{MobileNetV2}")   # never "Mo-bileNetV2"
        bad = [ch for ch in s if ord(ch) > 127]
        if bad:
            raise SystemExit(f"unmapped characters {set(bad)} in: {s[:120]}")
        for _ in range(3):   # links can hold code spans
            s = re.sub("\x00(\\d+)\x00", lambda m: store[int(m.group(1))], s)
        return s

    # ---------- blocks ----------
    def heading(self, level: int, text: str) -> str:
        if text == "References":
            return ("\n\\nocite{" + ",".join(NOCITE) + "}\n"
                    "\\printbibliography[heading=ctbib,title=References]\n")
        m = re.match(r"Appendix ([A-Z])\. (.+)", text)
        if m:
            out = ""
            if not self.appendix:
                self.appendix = True
                self.sec = [0, 0]
                out = "\n\\appendix\n"
            self.sec[0] += 1
            assert chr(64 + self.sec[0]) == m.group(1), text
            if m.group(1) == "A":
                out += "\\setcounter{table}{0}\\renewcommand{\\thetable}{A\\arabic{table}}\n"
                self.tables = 0
            return out + f"\n\\clearpage\n\\section{{{self.inline(m.group(2))}}}\n"
        if level == 2:
            m = re.match(r"(\d+)\. (.+)", text)
            self.sec = [self.sec[0] + 1, 0]
            assert m and int(m.group(1)) == self.sec[0], text
            return f"\n\\section{{{self.inline(m.group(2))}}}\n"
        if level == 3:
            m = re.match(r"(\d+)\.(\d+) (.+)", text)
            self.sec[1] += 1
            assert m and (int(m.group(1)), int(m.group(2))) == tuple(self.sec), text
            return f"\n\\subsection{{{self.inline(m.group(3))}}}\n"
        return f"\n\\subsubsection*{{{self.inline(text)}}}\n"

    def table(self, rows: list, caption: str | None) -> str:
        cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
        header, align, body = cells[0], cells[1], cells[2:]
        n = len(header)
        assert all(len(r) == n for r in body), rows[0]
        right = [a.endswith(":") and not a.startswith(":") for a in align]

        def plain(t: str, words: bool = False) -> str:
            t = re.sub(r"\]\([^)]*\)|\[", "", t)
            if words:   # LaTeX may break code spans at / . _ and any word at an explicit hyphen
                t = re.sub(r"`([^`]*)`", lambda m: re.sub(r"[/._]", " ", m.group(1)), t)
                t = t.replace("-", " ")
            return t.replace("`", "").replace("*", "")

        maxlen = [max(len(plain(r[j])) for r in [header] + body) for j in range(n)]
        word = [max((len(w) for r in [header] + body for w in plain(r[j], True).split()), default=1)
                for j in range(n)]
        size, widths = None, None
        for size in ([r"\small"] if n <= 3 else []) + [r"\footnotesize", r"\scriptsize"]:
            cw = CHAR_PT[size]
            need = [cw * maxlen[j] + 2 * TABCOLSEP for j in range(n)]
            mins = [max(cw * word[j] + 6, 24) + 2 * TABCOLSEP for j in range(n)]   # capitals run wider
            if sum(mins) > LINEWIDTH:
                continue
            if sum(need) <= LINEWIDTH:
                extra = LINEWIDTH - sum(need)
                widths = [need[j] + extra * need[j] / sum(need) for j in range(n)]
            else:
                flex = [need[j] - mins[j] for j in range(n)]
                rest = LINEWIDTH - sum(mins)
                widths = [mins[j] + rest * flex[j] / sum(flex) for j in range(n)]
            break
        if widths is None:          # last resort: scale the minimum widths down, words may hyphenate
            widths = [m * LINEWIDTH / sum(mins) for m in mins]
        spec = "".join(
            (r">{\raggedleft\arraybackslash}" if right[j] else r">{\raggedright\arraybackslash\hspace{0pt}}")
            + rf"p{{{widths[j] - 2 * TABCOLSEP:.1f}pt}}" for j in range(n))
        caption_tex = None
        if caption:
            m = re.match(r"Table (A?\d+)\. (.+)", caption)
            self.tables += 1
            expect = f"A{self.tables}" if self.appendix else str(self.tables)
            assert m and m.group(1) == expect, (caption[:40], expect)
            caption_tex = r"\caption{" + self.inline(m.group(2)) + "}"
        cw = CHAR_PT[size]
        lines = sum(max(-(-len(plain(c)) * cw // max(widths[j] - 2 * TABCOLSEP, 1)) for j, c in enumerate(r))
                    for r in body)
        # Short tables stay on one page; taller ones may split, but never right after the
        # first two rows or before the last two. The bottom rule and caption stay with the
        # last page (\endlastfoot), so a caption is never left alone on a page.
        keep = lambda k: lines <= 22 or k < 2 or k >= len(body) - 2
        out = ["", "{" + size, rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", r"\toprule",
               " & ".join(self.inline(h) for h in header) + r" \\", r"\midrule", r"\endhead",
               r"\bottomrule", r"\endfoot", r"\bottomrule"]
        out += [caption_tex] if caption_tex else []
        out.append(r"\endlastfoot")
        out += [" & ".join(self.inline(c) for c in r) + (r" \\*" if keep(k) else r" \\")
                for k, r in enumerate(body)]
        out.append(r"\end{longtable}}")
        if not caption_tex:
            out.append(r"\addtocounter{table}{-1}")
        return "\n".join(out) + "\n"

    def figure(self, target: str, caption: str) -> str:
        m = re.match(r"Figure (\d+)\. (.+)", caption)
        self.figures += 1
        assert m and int(m.group(1)) == self.figures, caption[:40]
        path = os.path.relpath((self.src.parent / target).resolve(), OUT_DIR).replace("\\", "/")
        return ("\n\\begin{figure}[H]\n\\centering\n"
                f"\\includegraphics[width=\\linewidth,height=0.66\\textheight,keepaspectratio]{{{path}}}\n"
                f"\\caption{{{self.inline(m.group(2))}}}\n\\end{{figure}}\n")

    def code_block(self, lines: list) -> str:
        text = "\n".join(lines)
        text = (text.replace("├── ", "|-- ").replace("└── ", "`-- ").replace("│", "|").replace("─", "-")
                .replace("…", "..."))
        bad = {ch for ch in text if ord(ch) > 127}
        assert not bad, bad
        return ("\n\\begin{Verbatim}[fontsize=\\fontsize{6.5}{8}\\selectfont,frame=none]\n"
                + text + "\n\\end{Verbatim}\n")

    def convert(self) -> str:
        meta = {}
        i = 0
        while not self.lines[i].startswith("## "):
            m = re.match(r"\*\*(\w+):\*\* (.+?)(?:<br>)?$", self.lines[i])
            if m:
                meta[m.group(1)] = m.group(2)
            i += 1
        student = re.search(r"\((\d+)\)", meta["Student"]).group(1)
        head = PREAMBLE
        for k, v in {"title": self.inline(meta["Project"]), "student": student,
                     "supervisor": self.inline(meta["Supervisor"]).replace("Dr. ", r"Dr.\ "),
                     "date": self.inline(meta["Date"]), "repo": meta["Repository"]}.items():
            head = head.replace(f"@@{k}@@", v)
        body = [FRONT]
        in_refs = False
        lines = self.lines
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                text = line[level:].strip()
                in_refs = text == "References"
                body.append(self.heading(level, text))
                i += 1
                continue
            if in_refs:                       # the reference list comes from refs.bib
                i += 1
                continue
            if line.startswith("```"):
                j = i + 1
                while not lines[j].startswith("```"):
                    j += 1
                body.append(self.code_block(lines[i + 1:j]))
                i = j + 1
                continue
            if line.startswith("|"):
                end = i
                while end < len(lines) and lines[end].startswith("|"):
                    end += 1
                k = end
                while k < len(lines) and not lines[k].strip():
                    k += 1
                caption = None
                if k < len(lines) and lines[k].startswith("*Table") and lines[k].endswith("*"):
                    caption = lines[k][1:-1]
                body.append(self.table(lines[i:end], caption))
                i = k + 1 if caption else end
                continue
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if m:
                k = i + 1
                while not lines[k].strip():
                    k += 1
                assert lines[k].startswith("*Figure"), lines[k][:40]
                body.append(self.figure(m.group(1), lines[k][1:-1]))
                i = k + 1
                continue
            if re.match(r"(- |\d+\. )", line):
                ordered = bool(re.match(r"\d+\. ", line))
                env = "enumerate" if ordered else "itemize"
                items = []
                while i < len(lines) and re.match(r"(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                    i += 1
                body.append(f"\n\\begin{{{env}}}\n" + "\n".join(r"\item " + self.inline(t) for t in items)
                            + f"\n\\end{{{env}}}\n")
                continue
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", "![")):
                para.append(lines[i])
                i += 1
            body.append("\n" + self.inline(" ".join(para)) + "\n")
        body.append("\n\\end{document}\n")
        return head + "".join(body)


def compile_pdf() -> None:
    def run(*cmd):
        r = subprocess.run([str(TEXBIN / cmd[0]), *cmd[1:]], cwd=OUT_DIR, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        if r.returncode != 0:
            tail = "\n".join((r.stdout or "").splitlines()[-25:])
            raise SystemExit(f"{cmd[0]} failed:\n{tail}")

    tex = ["pdflatex.exe", "-interaction=nonstopmode", "-halt-on-error", TEX.name]
    run(*tex)
    run("biber.exe", TEX.stem)
    run(*tex)
    run(*tex)
    log = (OUT_DIR / f"{TEX.stem}.log").read_text(encoding="utf-8", errors="replace")
    problems = [l for l in log.splitlines()
                if re.search(r"undefined|Missing character|Rerun to get", l, flags=re.I)]
    if problems:
        raise SystemExit("LaTeX problems:\n" + "\n".join(problems[:20]))
    shutil.copyfile(OUT_DIR / f"{TEX.stem}.pdf", PDF)
    print(f"wrote {PDF.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true", help="only write the .tex")
    args = ap.parse_args()
    conv = Converter(SRC)
    TEX.write_text(conv.convert(), encoding="utf-8")
    print(f"wrote {TEX.relative_to(ROOT)}: {conv.tables} captioned appendix/body tables in the last "
          f"numbering, {conv.figures} figures, {len(conv.cited)} cited keys")
    if not args.no_pdf:
        compile_pdf()


if __name__ == "__main__":
    main()
