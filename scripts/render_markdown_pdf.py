import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def escape_inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='Courier'>\1</font>", text)
    return text


def parse_markdown(md_text: str):
    lines = md_text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue

        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i].rstrip())
                i += 1
            blocks.append(("table", table_lines))
            continue

        if re.match(r"^#{1,3}\s+", line):
            level = len(line) - len(line.lstrip("#"))
            text = line[level:].strip()
            blocks.append(("heading", level, text))
            i += 1
            continue

        if re.match(r"^- ", line):
            items = []
            while i < len(lines) and re.match(r"^- ", lines[i].rstrip()):
                items.append(lines[i].rstrip()[2:].strip())
                i += 1
            blocks.append(("bullet", items))
            continue

        paragraph_lines = [line.strip()]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if not nxt.strip():
                i += 1
                break
            if nxt.startswith("|") or re.match(r"^#{1,3}\s+", nxt) or re.match(r"^- ", nxt):
                break
            paragraph_lines.append(nxt.strip())
            i += 1
        blocks.append(("paragraph", " ".join(paragraph_lines)))

    return blocks


def table_from_lines(table_lines, styles):
    rows = []
    for idx, raw in enumerate(table_lines):
        if idx == 1 and set(raw.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
            continue
        cells = [
            Paragraph(escape_inline(cell.strip()), styles["PaperBody"])
            for cell in raw.strip("|").split("|")
        ]
        rows.append(cells)

    table = Table(rows, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6E6E6")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8F8F8")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def build_pdf(input_path: Path, output_path: Path):
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="PaperTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperH1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperH2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="PaperBody",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            spaceAfter=6,
        )
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=input_path.stem,
    )

    story = []
    blocks = parse_markdown(input_path.read_text(encoding="utf-8"))

    for idx, block in enumerate(blocks):
        kind = block[0]
        if kind == "heading":
            _, level, text = block
            if idx == 0 and level == 1:
                style = styles["PaperTitle"]
            elif level == 1:
                style = styles["PaperH1"]
            else:
                style = styles["PaperH2"]
            story.append(Paragraph(escape_inline(text), style))
            if idx == 0:
                story.append(Spacer(1, 4))
        elif kind == "paragraph":
            _, text = block
            story.append(Paragraph(escape_inline(text), styles["PaperBody"]))
        elif kind == "bullet":
            _, items = block
            flowable_items = [
                ListItem(Paragraph(escape_inline(item), styles["PaperBody"]), leftIndent=8)
                for item in items
            ]
            story.append(ListFlowable(flowable_items, bulletType="bullet", start="circle", leftIndent=12))
            story.append(Spacer(1, 4))
        elif kind == "table":
            _, table_lines = block
            story.append(table_from_lines(table_lines, styles))
            story.append(Spacer(1, 8))

    doc.build(story)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_pdf(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
