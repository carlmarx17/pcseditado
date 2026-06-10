#!/usr/bin/env python3
import re
import sys
import textwrap
from pathlib import Path


PAGE_W = 595.28
PAGE_H = 841.89
MARGIN_L = 56
MARGIN_R = 56
MARGIN_T = 56
MARGIN_B = 54

FONT_NORMAL = "F1"
FONT_BOLD = "F2"
FONT_MONO = "F3"


def esc(text):
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def strip_inline(text):
    text = text.replace("`", "")
    text = text.replace("**", "")
    return text


def wrap_text(text, width):
    if not text:
        return [""]
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False) or [""]


class PdfBuilder:
    def __init__(self):
        self.pages = []
        self.lines = []
        self.y = PAGE_H - MARGIN_T

    def new_page(self):
        if self.lines:
            self.pages.append(self.lines)
        self.lines = []
        self.y = PAGE_H - MARGIN_T

    def ensure(self, amount):
        if self.y - amount < MARGIN_B:
            self.new_page()

    def add_line(self, text, size=10.5, font=FONT_NORMAL, leading=None, gap_after=0):
        leading = leading or size * 1.35
        self.ensure(leading + gap_after)
        self.lines.append((font, size, MARGIN_L, self.y, text))
        self.y -= leading + gap_after

    def add_gap(self, amount):
        self.ensure(amount)
        self.y -= amount

    def finish(self):
        if self.lines:
            self.pages.append(self.lines)
        return build_pdf(self.pages)


def markdown_to_pages(md_text):
    pdf = PdfBuilder()
    in_code = False
    paragraph = []

    def flush_paragraph():
        nonlocal paragraph
        if not paragraph:
            return
        text = strip_inline(" ".join(paragraph).strip())
        for line in wrap_text(text, 92):
            pdf.add_line(line, size=10.3)
        pdf.add_gap(5)
        paragraph = []

    for raw in md_text.splitlines():
        line = raw.rstrip()

        if line.strip().startswith("```"):
            flush_paragraph()
            in_code = not in_code
            if not in_code:
                pdf.add_gap(4)
            continue

        if in_code:
            for chunk in wrap_text(line, 82):
                pdf.add_line(chunk, size=8.8, font=FONT_MONO, leading=11)
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if line.startswith("# "):
            flush_paragraph()
            heading = strip_inline(line[2:].strip())
            for wrapped in wrap_text(heading, 54):
                pdf.add_line(wrapped, size=18, font=FONT_BOLD, leading=23)
            pdf.add_gap(8)
            continue

        if line.startswith("## "):
            flush_paragraph()
            pdf.add_gap(5)
            heading = strip_inline(line[3:].strip())
            for wrapped in wrap_text(heading, 68):
                pdf.add_line(wrapped, size=13.5, font=FONT_BOLD, leading=18)
            pdf.add_gap(3)
            continue

        if line.startswith("### "):
            flush_paragraph()
            pdf.add_gap(3)
            heading = strip_inline(line[4:].strip())
            for wrapped in wrap_text(heading, 78):
                pdf.add_line(wrapped, size=11.5, font=FONT_BOLD, leading=15)
            pdf.add_gap(2)
            continue

        if line.startswith("|"):
            flush_paragraph()
            if re.match(r"^\|[-:| ]+\|$", line):
                continue
            clean = strip_inline("  ".join(part.strip() for part in line.strip("|").split("|")))
            for chunk in wrap_text(clean, 78):
                pdf.add_line(chunk, size=8.4, font=FONT_MONO, leading=10.8)
            continue

        if line.startswith("- "):
            flush_paragraph()
            text = strip_inline(line[2:].strip())
            wrapped = wrap_text(text, 86)
            pdf.add_line("- " + wrapped[0], size=10.1)
            for extra in wrapped[1:]:
                pdf.add_line("  " + extra, size=10.1)
            continue

        ordered = re.match(r"^(\d+)\.\s+(.*)$", line)
        if ordered:
            flush_paragraph()
            prefix = ordered.group(1) + ". "
            wrapped = wrap_text(strip_inline(ordered.group(2)), 84)
            pdf.add_line(prefix + wrapped[0], size=10.1)
            for extra in wrapped[1:]:
                pdf.add_line("   " + extra, size=10.1)
            continue

        paragraph.append(line)

    flush_paragraph()
    return pdf.finish()


def build_pdf(pages):
    objects = []

    def add(obj):
        objects.append(obj)
        return len(objects)

    catalog_id = add("<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add("")
    font1_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    font2_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
    font3_id = add("<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")

    page_ids = []
    for page in pages:
        stream_lines = ["BT"]
        current_font = None
        current_size = None
        for font, size, x, y, text in page:
            if (font, size) != (current_font, current_size):
                stream_lines.append(f"/{font} {size:.2f} Tf")
                current_font, current_size = font, size
            stream_lines.append(f"1 0 0 1 {x:.2f} {y:.2f} Tm ({esc(text)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", "replace")
        content_id = add(f"<< /Length {len(stream)} >>\nstream\n{stream.decode('latin-1')}\nendstream")
        page_id = add(
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_W:.2f} {PAGE_H:.2f}] "
            f"/Resources << /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R /F3 {font3_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>"
        )
        page_ids.append(page_id)

    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {len(page_ids)} >>"

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin-1", "replace"))
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(out)


def main():
    if len(sys.argv) != 3:
        print("usage: md_to_simple_pdf.py input.md output.pdf", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    target = Path(sys.argv[2])
    target.write_bytes(markdown_to_pages(source.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
