"""Safe Markdown, DOCX, and XML-escaped ReportLab PDF report generation."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def create_markdown(title: str, content: str, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{content}", encoding="utf-8")
    return str(path)


def create_pdf(title: str, content: str, output_path: str | Path) -> str:
    """Render Markdown-like text to PDF with XML escaping so ReportLab never fails on < or &."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1, 12)]

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 6))
            continue
        safe = escape(line)
        if safe.startswith("## "):
            story.append(Paragraph(safe[3:], styles["Heading2"]))
        elif safe.startswith("# "):
            story.append(Paragraph(safe[2:], styles["Heading1"]))
        elif safe.startswith("- "):
            story.append(Paragraph(f"• {safe[2:]}", styles["BodyText"]))
        else:
            story.append(Paragraph(safe, styles["BodyText"]))

    doc.build(story)
    return str(path)


def create_docx(title: str, content: str, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading(title, level=1)
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        elif stripped:
            doc.add_paragraph(stripped)
    doc.save(str(path))
    return str(path)
