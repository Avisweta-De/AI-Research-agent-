"""
PDF Report Generator
Converts a markdown research report into a professionally styled PDF
using ReportLab's Platypus engine.
"""

from __future__ import annotations

import os
import re
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    HRFlowable,
    Table,
    TableStyle,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
PRIMARY = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#4361ee")
ACCENT_LIGHT = colors.HexColor("#7b8ff7")
TEXT_DARK = colors.HexColor("#1a1a2e")
TEXT_MEDIUM = colors.HexColor("#4a4a5a")
TEXT_LIGHT = colors.HexColor("#6b6b7b")
BG_LIGHT = colors.HexColor("#f0f2f5")
BORDER = colors.HexColor("#d0d5dd")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("output")


def _get_styles():
    """Build a custom stylesheet for the research report."""
    styles = getSampleStyleSheet()

    # Cover title
    styles.add(ParagraphStyle(
        name="CoverTitle",
        parent=styles["Title"],
        fontSize=28,
        leading=34,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=20,
        fontName="Helvetica-Bold",
    ))

    # Cover subtitle
    styles.add(ParagraphStyle(
        name="CoverSubtitle",
        parent=styles["Normal"],
        fontSize=14,
        leading=18,
        textColor=TEXT_MEDIUM,
        alignment=TA_CENTER,
        spaceAfter=8,
    ))

    # Section heading (H1) — e.g. "# Executive Summary"
    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading1"],
        fontSize=20,
        leading=26,
        textColor=PRIMARY,
        spaceBefore=24,
        spaceAfter=12,
        fontName="Helvetica-Bold",
        borderWidth=0,
        borderPadding=0,
    ))

    # Subsection heading (H2) — e.g. "## Subtopic"
    styles.add(ParagraphStyle(
        name="SubsectionHeading",
        parent=styles["Heading2"],
        fontSize=15,
        leading=20,
        textColor=ACCENT,
        spaceBefore=16,
        spaceAfter=8,
        fontName="Helvetica-Bold",
    ))

    # Body text
    styles.add(ParagraphStyle(
        name="BodyText2",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=TEXT_DARK,
        alignment=TA_JUSTIFY,
        spaceAfter=8,
        fontName="Helvetica",
    ))

    # Bullet item
    styles.add(ParagraphStyle(
        name="BulletItem",
        parent=styles["Normal"],
        fontSize=10.5,
        leading=15,
        textColor=TEXT_DARK,
        leftIndent=20,
        spaceAfter=4,
        bulletIndent=8,
        fontName="Helvetica",
    ))

    # Citation / reference entry
    styles.add(ParagraphStyle(
        name="Reference",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=TEXT_MEDIUM,
        leftIndent=24,
        firstLineIndent=-24,
        spaceAfter=4,
        fontName="Helvetica",
    ))

    # Footer
    styles.add(ParagraphStyle(
        name="Footer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=TEXT_LIGHT,
        alignment=TA_CENTER,
    ))

    return styles


def _add_header_footer(canvas, doc):
    """Draw header line and footer on each page."""
    canvas.saveState()
    width, height = A4

    # Header accent line
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2)
    canvas.line(40, height - 40, width - 40, height - 40)

    # Footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(TEXT_LIGHT)
    canvas.drawCentredString(
        width / 2,
        25,
        f"AI-Generated Research Report  •  Page {doc.page}  •  {datetime.now().strftime('%B %d, %Y')}",
    )

    canvas.restoreState()


def _markdown_to_flowables(markdown_text: str, styles) -> list:
    """
    Convert markdown text into ReportLab Platypus flowables.
    Handles: headings, paragraphs, bullet lists, bold/italic, citations.
    """
    flowables = []
    lines = markdown_text.split("\n")
    i = 0
    paragraph_buffer = []

    def flush_paragraph():
        if paragraph_buffer:
            text = " ".join(paragraph_buffer)
            text = _format_inline(text)
            flowables.append(Paragraph(text, styles["BodyText2"]))
            paragraph_buffer.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip empty lines — flush paragraph
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        # H1: # Heading
        if stripped.startswith("# ") and not stripped.startswith("## "):
            flush_paragraph()
            heading = _format_inline(stripped[2:].strip())
            flowables.append(Spacer(1, 6))
            flowables.append(HRFlowable(
                width="100%", thickness=1, color=BORDER,
                spaceBefore=4, spaceAfter=4,
            ))
            flowables.append(Paragraph(heading, styles["SectionHeading"]))
            i += 1
            continue

        # H2: ## Heading
        if stripped.startswith("## "):
            flush_paragraph()
            heading = _format_inline(stripped[3:].strip())
            flowables.append(Paragraph(heading, styles["SubsectionHeading"]))
            i += 1
            continue

        # H3+: ### Heading (treated as bold paragraph)
        if stripped.startswith("### "):
            flush_paragraph()
            heading = _format_inline(stripped[4:].strip())
            flowables.append(Paragraph(f"<b>{heading}</b>", styles["BodyText2"]))
            i += 1
            continue

        # Bullet: - item or * item
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_paragraph()
            text = _format_inline(stripped[2:].strip())
            flowables.append(Paragraph(f"•  {text}", styles["BulletItem"]))
            i += 1
            continue

        # Numbered list: 1. item
        num_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if num_match:
            flush_paragraph()
            num = num_match.group(1)
            text = _format_inline(num_match.group(2))
            flowables.append(Paragraph(f"{num}. {text}", styles["BulletItem"]))
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            flush_paragraph()
            flowables.append(HRFlowable(
                width="100%", thickness=0.5, color=BORDER,
                spaceBefore=8, spaceAfter=8,
            ))
            i += 1
            continue

        # Regular text → accumulate into paragraph
        paragraph_buffer.append(stripped)
        i += 1

    flush_paragraph()
    return flowables


def _format_inline(text: str) -> str:
    """Convert markdown inline formatting to ReportLab XML tags."""
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)

    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)

    # Inline code: `code`
    text = re.sub(r"`(.+?)`", r'<font face="Courier" size="9">\1</font>', text)

    # Links: [text](url) → text (url)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'\1 (<font color="#4361ee">\2</font>)', text)

    # Citation brackets: [1] → styled
    text = re.sub(
        r"\[(\d+)\]",
        r'<font color="#4361ee"><super>[\1]</super></font>',
        text,
    )

    # Escape any remaining XML-unsafe characters
    # (but don't double-escape our tags)
    text = text.replace("&", "&amp;").replace("&amp;amp;", "&amp;")

    return text


def generate_pdf(
    report_markdown: str,
    topic: str,
    output_dir: str | Path | None = None,
) -> str:
    """
    Generate a styled PDF report from markdown text.

    Args:
        report_markdown: The full report in markdown format.
        topic: The research topic (for the cover page).
        output_dir: Directory to save the PDF. Defaults to ./output/.

    Returns:
        Absolute path to the generated PDF file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename from topic
    safe_name = re.sub(r"[^\w\s-]", "", topic)[:60].strip().replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"research_report_{safe_name}_{timestamp}.pdf"
    filepath = output_dir / filename

    styles = _get_styles()

    doc = SimpleDocTemplate(
        str(filepath),
        pagesize=A4,
        topMargin=55,
        bottomMargin=45,
        leftMargin=50,
        rightMargin=50,
        title=f"Research Report: {topic}",
        author="Multi-Agent Research System",
    )

    story: list = []

    # ── Cover page ──────────────────────────────────────────────────────
    story.append(Spacer(1, 2 * inch))
    story.append(Paragraph("RESEARCH REPORT", styles["CoverSubtitle"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(topic, styles["CoverTitle"]))
    story.append(Spacer(1, 24))
    story.append(HRFlowable(
        width="40%", thickness=2, color=ACCENT,
        spaceBefore=0, spaceAfter=16, hAlign="CENTER",
    ))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
        styles["CoverSubtitle"],
    ))
    story.append(Paragraph(
        "Multi-Agent AI Research System  •  Autonomous Report Generation",
        styles["CoverSubtitle"],
    ))
    story.append(Spacer(1, 1 * inch))

    # Metadata box
    meta_data = [
        ["Framework", "LangGraph Multi-Agent Pipeline"],
        ["LLM", "Llama 3.3 70B (via Groq)"],
        ["Sources", "Tavily Web Search + ArXiv Academic Papers"],
        ["Report Type", "Comprehensive Research Synthesis"],
    ]
    meta_table = Table(meta_data, colWidths=[140, 300])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), TEXT_MEDIUM),
        ("TEXTCOLOR", (1, 0), (1, -1), TEXT_DARK),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(meta_table)
    story.append(PageBreak())

    # ── Report body ─────────────────────────────────────────────────────
    body_flowables = _markdown_to_flowables(report_markdown, styles)
    story.extend(body_flowables)

    # ── Build the PDF ───────────────────────────────────────────────────
    try:
        doc.build(story, onFirstPage=_add_header_footer, onLaterPages=_add_header_footer)
        logger.info("PDF generated: %s", filepath)
    except Exception as e:
        logger.error("PDF generation failed: %s", e)
        raise

    return str(filepath.resolve())
