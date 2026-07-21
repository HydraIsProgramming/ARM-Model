"""Convert PROGRESS_REPORT_MAY_JUNE_2026.md to a formatted PDF using reportlab."""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

MD = Path(__file__).parent.parent / "docs" / "PROGRESS_REPORT_MAY_JUNE_2026.md"
OUT = Path(__file__).parent.parent / "docs" / "PROGRESS_REPORT_MAY_JUNE_2026.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

BRAND_BLUE  = colors.HexColor("#1a3a5c")
BRAND_GOLD  = colors.HexColor("#c8a951")
LIGHT_GRAY  = colors.HexColor("#f5f5f5")
MID_GRAY    = colors.HexColor("#cccccc")
CODE_BG     = colors.HexColor("#f0f0f0")
WHITE       = colors.white

title_style = ParagraphStyle(
    "Title", parent=styles["Normal"],
    fontSize=22, leading=28, textColor=BRAND_BLUE,
    fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "Subtitle", parent=styles["Normal"],
    fontSize=11, leading=15, textColor=colors.HexColor("#555555"),
    fontName="Helvetica", spaceAfter=2, alignment=TA_CENTER,
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Normal"],
    fontSize=14, leading=18, textColor=WHITE,
    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6,
    leftIndent=0, backColor=BRAND_BLUE, borderPad=5,
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Normal"],
    fontSize=12, leading=16, textColor=BRAND_BLUE,
    fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=4,
    leftIndent=0,
)
h3_style = ParagraphStyle(
    "H3", parent=styles["Normal"],
    fontSize=10, leading=14, textColor=BRAND_BLUE,
    fontName="Helvetica-BoldOblique", spaceBefore=8, spaceAfter=3,
)
body_style = ParagraphStyle(
    "Body", parent=styles["Normal"],
    fontSize=9.5, leading=14, textColor=colors.HexColor("#222222"),
    fontName="Helvetica", spaceAfter=4,
)
bullet_style = ParagraphStyle(
    "Bullet", parent=body_style,
    leftIndent=16, firstLineIndent=-10, spaceAfter=2,
    bulletIndent=6,
)
code_style = ParagraphStyle(
    "Code", parent=styles["Normal"],
    fontSize=8, leading=11, textColor=colors.HexColor("#333333"),
    fontName="Courier", backColor=CODE_BG,
    leftIndent=12, rightIndent=12, spaceBefore=4, spaceAfter=4,
    borderPad=4,
)


def make_table(headers, rows):
    col_count = len(headers)
    col_width = (7.0 * inch) / col_count
    col_widths = [col_width] * col_count

    table_data = [headers] + rows
    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  BRAND_BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  WHITE),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  9),
        ("BOTTOMPADDING",(0, 0), (-1, 0),  6),
        ("TOPPADDING",   (0, 0), (-1, 0),  6),
        ("FONTNAME",     (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("ROWBACKGROUNDS",(0, 1),(-1, -1), [WHITE, LIGHT_GRAY]),
        ("GRID",         (0, 0), (-1, -1), 0.4, MID_GRAY),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",   (0, 1), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 1), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def parse_md_to_flowables(md_text):
    flowables = []
    lines = md_text.splitlines()
    i = 0
    in_table = False
    table_headers = []
    table_rows = []
    in_code = False
    code_lines = []

    while i < len(lines):
        line = lines[i]

        # ── Code block ──────────────────────────────────────────────────
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                code_text = "\n".join(code_lines)
                flowables.append(Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_style))
                flowables.append(Spacer(1, 4))
            i += 1
            continue
        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # ── Table ────────────────────────────────────────────────────────
        if line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_line.startswith("|---") or next_line.startswith("| ---"):
                table_headers = cells
                in_table = True
                i += 2  # skip separator
                continue
            if in_table:
                table_rows.append(cells)
                i += 1
                continue
        else:
            if in_table and table_rows:
                flowables.append(Spacer(1, 4))
                flowables.append(make_table(table_headers, table_rows))
                flowables.append(Spacer(1, 6))
            in_table = False
            table_headers = []
            table_rows = []

        stripped = line.strip()

        # ── Headings ─────────────────────────────────────────────────────
        if stripped.startswith("# ") and not stripped.startswith("## "):
            text = stripped[2:].strip()
            flowables.append(Spacer(1, 8))
            flowables.append(Paragraph(text, h1_style))
            flowables.append(Spacer(1, 4))

        elif stripped.startswith("## "):
            text = stripped[3:].strip()
            flowables.append(Spacer(1, 6))
            flowables.append(HRFlowable(width="100%", thickness=1.5, color=BRAND_GOLD, spaceAfter=3))
            flowables.append(Paragraph(text, h2_style))

        elif stripped.startswith("### "):
            text = stripped[4:].strip()
            flowables.append(Paragraph(text, h3_style))

        # ── Horizontal rule ───────────────────────────────────────────────
        elif stripped.startswith("---"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceBefore=4, spaceAfter=4))

        # ── Bullets ───────────────────────────────────────────────────────
        elif stripped.startswith("- ") or stripped.startswith("* "):
            text = stripped[2:].strip()
            text = _escape(text)
            flowables.append(Paragraph(f"• {text}", bullet_style))

        # ── Empty line ────────────────────────────────────────────────────
        elif stripped == "":
            flowables.append(Spacer(1, 4))

        # ── Regular paragraph ─────────────────────────────────────────────
        else:
            text = _escape(stripped)
            flowables.append(Paragraph(text, body_style))

        i += 1

    # flush any trailing table
    if in_table and table_rows:
        flowables.append(make_table(table_headers, table_rows))

    return flowables


def _escape(text):
    """Basic markdown inline → reportlab XML conversion."""
    import re
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    # Inline code
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8">\1</font>', text)
    # Escape bare & that aren't already entities
    text = re.sub(r"&(?!#?\w+;)", "&amp;", text)
    return text


def build_pdf():
    md_text = MD.read_text(encoding="utf-8")

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=1.0 * inch,
        bottomMargin=0.85 * inch,
        title="CP493 Progress Report — Ranjot Sandhu",
        author="Ranjot Sandhu",
    )

    story = []

    # Cover block
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("CP493 Directed Research", title_style))
    story.append(Paragraph("Progress Report: May 1 – June 22, 2026", subtitle_style))
    story.append(Paragraph("Ranjot Sandhu · Wilfrid Laurier University", subtitle_style))
    story.append(Paragraph("RL Control of a 2-DOF Robotic Arm (Fischer et al. 2021)", subtitle_style))
    story.append(Spacer(1, 0.1 * inch))
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_GOLD, spaceBefore=6, spaceAfter=12))

    # Skip the first 4 lines of the MD (the frontmatter title block we already rendered)
    lines = md_text.splitlines()
    trimmed = "\n".join(lines[6:])  # skip title, bold lines, blank lines at top

    story += parse_md_to_flowables(trimmed)

    doc.build(story)
    print(f"PDF written to: {OUT}")


if __name__ == "__main__":
    build_pdf()
