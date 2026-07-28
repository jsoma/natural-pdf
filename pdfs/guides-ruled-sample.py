"""Generate pdfs/guides-ruled-sample.pdf — synthetic fixture for the Guides tutorial.

Run with: uv run --with reportlab python temp/make_guides_ruled_sample.py

One letter page, two tables:

1. "Field Equipment Purchases" — a fully ruled 5-column table (vector lines for
   every row and column boundary). Vendor names are underlined with short drawn
   lines, so naive horizontal line detection picks up 8 spurious boundaries —
   motivates from_lines(n=...). A decorative rule under the page title sits
   outside the table region.

2. "Petty Cash — March" — a 3-column listing with no ruling lines at all,
   separated only by whitespace gaps. Amounts are left-aligned so they share a
   common x0 (used for the snap_to_content demo).
"""

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

OUT = "pdfs/guides-ruled-sample.pdf"

W, H = letter  # 612 x 792

c = canvas.Canvas(OUT, pagesize=letter)

# --- Title + decorative rule (outside any table) ---
c.setFont("Helvetica-Bold", 16)
c.drawString(50, 745, "Field Equipment Purchases")
c.setLineWidth(1.5)
c.line(50, 738, 562, 738)
c.setFont("Helvetica", 9)
c.drawString(50, 726, "Quarter 1 — approved requisitions")

# --- Table 1: ruled, 5 columns, header + 8 data rows ---
col_x = [50, 210, 330, 385, 465, 562]
top_y = 710
row_h = 24
n_rows = 9  # header + 8 data

rows = [
    ("Item", "Vendor", "Qty", "Unit Price", "Total"),
    ("Soil auger, 3in", "Forestry Direct", "2", "$84.00", "$168.00"),
    ("Sample bags (500)", "LabSource", "4", "$22.50", "$90.00"),
    ("GPS receiver", "TrailTech", "1", "$412.00", "$412.00"),
    ("Water test kit", "LabSource", "6", "$31.75", "$190.50"),
    ("Folding table", "Office Depot", "2", "$45.99", "$91.98"),
    ("Clipboards", "Office Depot", "12", "$3.25", "$39.00"),
    ("Waders, size L", "Forestry Direct", "3", "$97.00", "$291.00"),
    ("First aid kit", "SafetyPro", "2", "$28.40", "$56.80"),
]

bottom_y = top_y - n_rows * row_h

# Grid lines
c.setLineWidth(0.8)
for i in range(n_rows + 1):
    y = top_y - i * row_h
    c.line(col_x[0], y, col_x[-1], y)
for x in col_x:
    c.line(x, bottom_y, x, top_y)

# Cell text; vendor column underlined (hyperlink style)
for r, row in enumerate(rows):
    y_base = top_y - (r + 1) * row_h + 8
    c.setFont("Helvetica-Bold" if r == 0 else "Helvetica", 9)
    for col, text in enumerate(row):
        x = col_x[col] + 5
        c.drawString(x, y_base, text)
        if r > 0 and col == 1:  # underline vendor names
            width = c.stringWidth(text, "Helvetica", 9)
            c.setLineWidth(0.6)
            c.line(x, y_base - 1.5, x + width, y_base - 1.5)

# --- Table 2: no lines at all, 3 columns with whitespace gaps ---
c.setFont("Helvetica-Bold", 13)
c.drawString(50, 430, "Petty Cash — March")

petty = [
    ("03/02", "Parking, county records office", "$12.00"),
    ("03/05", "Copies and printing", "$8.40"),
    ("03/11", "Mileage reimbursement, site visit", "$44.16"),
    ("03/14", "Postage", "$5.80"),
    ("03/21", "Batteries for GPS units", "$19.98"),
    ("03/28", "Coffee for volunteer training", "$23.50"),
]

p_top = 405
p_row_h = 20
c.setFont("Helvetica", 10)
for r, (date, desc, amount) in enumerate(petty):
    y = p_top - r * p_row_h
    c.drawString(50, y, date)
    c.drawString(150, y, desc)
    c.drawString(470, y, amount)  # left-aligned: shared x0

c.save()
print(f"wrote {OUT}")
