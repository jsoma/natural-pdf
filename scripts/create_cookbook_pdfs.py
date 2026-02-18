#!/usr/bin/env python3
"""Create sample PDFs for cookbook documentation."""

import io
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUTPUT_DIR = Path(__file__).parent.parent / "pdfs" / "cookbook"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()


def create_quarterly_sales():
    """PDF 1: Simple Tables - quarterly sales report with clean table."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "quarterly_sales.pdf"),
        pagesize=letter,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=14,
        alignment=TA_CENTER,
        spaceAfter=30,
    )

    story = []

    # Title
    story.append(Paragraph("QUARTERLY SALES REPORT", title_style))
    story.append(Paragraph("Q3 2024", subtitle_style))
    story.append(Spacer(1, 20))

    # Table with borders
    data = [
        ["Product", "Units Sold", "Revenue"],
        ["Widget A", "1,234", "$45,678"],
        ["Widget B", "567", "$12,345"],
        ["Widget C", "890", "$23,456"],
        ["Widget D", "2,345", "$67,890"],
        ["TOTAL", "5,036", "$149,369"],
    ]

    table = Table(data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -2), colors.beige),
                ("BACKGROUND", (0, -1), (-1, -1), colors.lightgrey),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 30))

    # Notes
    notes_style = ParagraphStyle(
        "Notes",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
    )
    story.append(Paragraph("Notes: All figures are preliminary and subject to audit.", notes_style))

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'quarterly_sales.pdf'}")


def create_vendor_invoice():
    """PDF 2: Form with Key-Value Pairs - vendor invoice."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "vendor_invoice.pdf"),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    label_style = ParagraphStyle(
        "Label",
        parent=styles["Normal"],
        fontSize=11,
        fontName="Helvetica-Bold",
    )

    value_style = ParagraphStyle(
        "Value",
        parent=styles["Normal"],
        fontSize=11,
    )

    story = []

    # Header
    story.append(Paragraph("VENDOR INVOICE", title_style))
    story.append(Spacer(1, 10))

    # Invoice metadata as key-value pairs
    metadata = [
        ["Invoice Number:", "INV-2024-00789"],
        ["Invoice Date:", "2024-03-15"],
        ["Due Date:", "2024-04-15"],
        ["Vendor:", "Acme Corporation"],
        ["Vendor Address:", "123 Business Ave, Suite 100"],
        ["PO Number:", "PO-2024-456"],
    ]

    meta_table = Table(metadata, colWidths=[1.5 * inch, 3 * inch])
    meta_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                ("ALIGN", (1, 0), (1, -1), "LEFT"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(meta_table)
    story.append(Spacer(1, 20))

    # Line items table
    story.append(Paragraph("Line Items", label_style))
    story.append(Spacer(1, 10))

    line_items = [
        ["Description", "Quantity", "Unit Price", "Amount"],
        ["Consulting Services - Senior", "40 hrs", "$100.00", "$4,000.00"],
        ["Consulting Services - Junior", "20 hrs", "$75.00", "$1,500.00"],
        ["Software License (Annual)", "1", "$2,500.00", "$2,500.00"],
        ["Travel Expenses", "1", "$350.00", "$350.00"],
    ]

    items_table = Table(line_items, colWidths=[2.5 * inch, 1 * inch, 1 * inch, 1 * inch])
    items_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(items_table)
    story.append(Spacer(1, 20))

    # Totals
    totals = [
        ["", "", "Subtotal:", "$8,350.00"],
        ["", "", "Tax (8%):", "$668.00"],
        ["", "", "TOTAL:", "$9,018.00"],
    ]

    totals_table = Table(totals, colWidths=[2.5 * inch, 1 * inch, 1 * inch, 1 * inch])
    totals_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (2, 0), (2, -1), "RIGHT"),
                ("ALIGN", (3, 0), (3, -1), "RIGHT"),
                ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (2, -1), (-1, -1), 12),
                ("LINEABOVE", (2, -1), (-1, -1), 1, colors.black),
            ]
        )
    )
    story.append(totals_table)
    story.append(Spacer(1, 30))

    # Payment info
    payment_style = ParagraphStyle(
        "Payment",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.grey,
    )
    story.append(Paragraph("Payment Terms: Net 30 days", payment_style))
    story.append(
        Paragraph("Please remit payment to: Acme Corporation, Account #12345678", payment_style)
    )

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'vendor_invoice.pdf'}")


def create_facility_inspections():
    """PDF 3: Multi-Page Repeating Forms - facility inspection reports."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "facility_inspections.pdf"),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    page_num_style = ParagraphStyle(
        "PageNum",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20,
    )

    # Data for 3 facility inspections
    inspections = [
        {
            "facility_id": "FAC-001",
            "facility_name": "Downtown Restaurant",
            "inspection_date": "2024-01-15",
            "inspector": "J. Smith",
            "status": "PASS",
            "score": "92",
            "violations": [
                ["V-101", "Minor", "Handwashing sign missing", "Corrected on-site"],
                ["V-205", "Minor", "Food temp log incomplete", "Follow-up required"],
            ],
        },
        {
            "facility_id": "FAC-002",
            "facility_name": "Eastside Deli",
            "inspection_date": "2024-01-16",
            "inspector": "M. Johnson",
            "status": "FAIL",
            "score": "68",
            "violations": [
                ["V-301", "Critical", "Refrigerator temp above 41F", "Re-inspection needed"],
                ["V-102", "Major", "Employee illness policy not posted", "Pending"],
                ["V-205", "Minor", "Food temp log incomplete", "Pending"],
                ["V-401", "Major", "Pest evidence in storage area", "Pending"],
                ["V-103", "Minor", "Sanitizer concentration low", "Corrected on-site"],
            ],
        },
        {
            "facility_id": "FAC-003",
            "facility_name": "Westside Cafe",
            "inspection_date": "2024-01-17",
            "inspector": "J. Smith",
            "status": "PASS",
            "score": "98",
            "violations": [
                ["V-101", "Minor", "Handwashing sign faded", "Corrected on-site"],
            ],
        },
    ]

    story = []

    for i, insp in enumerate(inspections):
        if i > 0:
            story.append(PageBreak())

        # Header
        story.append(Paragraph("FACILITY INSPECTION REPORT", title_style))
        story.append(Paragraph(f"Page {i+1} of {len(inspections)}", page_num_style))

        # Facility info as 2-column key-value pairs (easier to extract with .right())
        info = [
            ["Facility ID:", insp["facility_id"]],
            ["Facility Name:", insp["facility_name"]],
            ["Inspection Date:", insp["inspection_date"]],
            ["Inspector:", insp["inspector"]],
            ["Status:", insp["status"]],
            ["Score:", insp["score"]],
        ]

        info_table = Table(info, colWidths=[1.5 * inch, 3 * inch])
        info_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("ALIGN", (0, 0), (0, -1), "RIGHT"),
                    ("ALIGN", (1, 0), (1, -1), "LEFT"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOX", (0, 0), (-1, -1), 1, colors.black),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    (
                        "BACKGROUND",
                        (1, 4),
                        (1, 4),
                        colors.lightgreen if insp["status"] == "PASS" else colors.lightcoral,
                    ),
                ]
            )
        )
        story.append(info_table)
        story.append(Spacer(1, 20))

        # Violations header
        section_style = ParagraphStyle(
            "Section",
            parent=styles["Heading2"],
            fontSize=12,
            spaceAfter=10,
        )
        story.append(Paragraph(f"Violations Found: {len(insp['violations'])}", section_style))

        # Violations table
        viol_header = ["Code", "Severity", "Description", "Status"]
        viol_data = [viol_header] + insp["violations"]

        viol_table = Table(viol_data, colWidths=[0.8 * inch, 0.8 * inch, 2.5 * inch, 1.4 * inch])
        viol_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )

        # Color code severity
        for row_idx, viol in enumerate(insp["violations"], start=1):
            severity = viol[1]
            if severity == "Critical":
                viol_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (1, row_idx), (1, row_idx), colors.lightcoral),
                        ]
                    )
                )
            elif severity == "Major":
                viol_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (1, row_idx), (1, row_idx), colors.lightyellow),
                        ]
                    )
                )

        story.append(viol_table)
        story.append(Spacer(1, 20))

        # Footer
        footer_style = ParagraphStyle(
            "Footer",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.grey,
        )
        story.append(Paragraph(f"Inspector Signature: {insp['inspector']}", footer_style))
        story.append(Paragraph("This report is official and confidential.", footer_style))

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'facility_inspections.pdf'}")


def create_incident_log():
    """PDF 4: Messy Table - incident log with multi-line cells and continuation rows."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "incident_log.pdf"),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    story = []
    story.append(Paragraph("INCIDENT LOG - January 2024", title_style))
    story.append(Spacer(1, 10))

    # Create messy table with multi-line descriptions and continuation rows
    # Using Paragraph objects to allow text wrapping
    cell_style = ParagraphStyle(
        "Cell",
        parent=styles["Normal"],
        fontSize=9,
        leading=11,
    )

    data = [
        ["ID", "Date", "Officer(s)", "Location", "Description"],
        [
            "001",
            "01/15",
            "Smith, J.",
            "100 Main St",
            Paragraph(
                "Traffic stop - vehicle observed running red light. Driver cited for violation. No further action.",
                cell_style,
            ),
        ],
        [
            "002",
            "01/15",
            "Smith, J.\nJohnson, M.",
            "250 Oak Ave",
            Paragraph(
                "Disturbance call at residential address. Parties separated and advised. One party left premises voluntarily. Follow-up scheduled.",
                cell_style,
            ),
        ],
        [
            "",
            "(continued)",
            "",
            "",
            Paragraph(
                "Additional notes: Neighbor reported ongoing disputes. Case referred to community mediation.",
                cell_style,
            ),
        ],
        [
            "003",
            "01/16",
            "Davis, R.",
            "500 Park Blvd",
            Paragraph(
                "Welfare check requested by family member. Subject located and in good health. No services needed.",
                cell_style,
            ),
        ],
        [
            "004",
            "01/16",
            "Davis, R.\nSmith, J.\nWilson, K.",
            "75 Commerce Dr",
            Paragraph(
                "Shoplifting report at retail store. Suspect apprehended by store security. Items valued at $156.00 recovered. Suspect transported to station.",
                cell_style,
            ),
        ],
        [
            "",
            "(continued)",
            "",
            "",
            Paragraph(
                "Suspect identified as J. Doe (DOB: 03/15/1990). Prior record: 2 misdemeanors. Released on citation, court date set for 02/15/2024.",
                cell_style,
            ),
        ],
        [
            "005",
            "01/17",
            "Johnson, M.",
            "300 River Rd",
            Paragraph(
                "Vehicle accident - two cars, minor damage. No injuries reported. Insurance information exchanged. Report filed.",
                cell_style,
            ),
        ],
        [
            "006",
            "01/17",
            "",
            "425 Industrial Way",
            Paragraph(
                "Anonymous tip received. Area checked, nothing found. Case closed.", cell_style
            ),
        ],
    ]

    col_widths = [0.5 * inch, 0.7 * inch, 1 * inch, 1.1 * inch, 3 * inch]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4057")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 10),
                ("FONTSIZE", (0, 1), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                # Highlight continuation rows
                ("BACKGROUND", (0, 3), (-1, 3), colors.HexColor("#F0F0F0")),
                ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#F0F0F0")),
                ("FONTNAME", (1, 3), (1, 3), "Helvetica-Oblique"),
                ("FONTNAME", (1, 6), (1, 6), "Helvetica-Oblique"),
                # Missing officer highlighting
                ("BACKGROUND", (2, 8), (2, 8), colors.HexColor("#FFEEEE")),
            ]
        )
    )

    story.append(table)
    story.append(Spacer(1, 20))

    notes_style = ParagraphStyle(
        "Notes",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    )
    story.append(
        Paragraph("Legend: Shaded rows indicate continuation from previous entry.", notes_style)
    )
    story.append(Paragraph("Pink cells indicate missing data requiring follow-up.", notes_style))

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'incident_log.pdf'}")


def create_budget_items():
    """PDF 6: Multi-Page Table - budget items spanning two pages."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "budget_items.pdf"),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=16,
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=12,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=20,
    )

    story = []

    # Page 1
    story.append(Paragraph("DEPARTMENT BUDGET - FY 2024", title_style))
    story.append(Paragraph("Information Technology Division", subtitle_style))

    # Header note
    note_style = ParagraphStyle(
        "Note",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=15,
    )
    story.append(
        Paragraph("Budget line items by category. Table continues on next page.", note_style)
    )

    # Budget data split across pages
    page1_data = [
        ["Line", "Category", "Description", "Amount"],
        ["1001", "Personnel", "Full-time salaries", "$450,000"],
        ["1002", "Personnel", "Part-time wages", "$85,000"],
        ["1003", "Personnel", "Benefits (health, retirement)", "$125,000"],
        ["1004", "Personnel", "Training and development", "$15,000"],
        ["2001", "Equipment", "Computers and laptops", "$75,000"],
        ["2002", "Equipment", "Servers and networking", "$120,000"],
        ["2003", "Equipment", "Printers and peripherals", "$12,000"],
        ["2004", "Equipment", "Mobile devices", "$18,000"],
        ["3001", "Software", "Enterprise licenses", "$95,000"],
        ["3002", "Software", "Cloud services (AWS/Azure)", "$65,000"],
        ["3003", "Software", "Security tools", "$35,000"],
    ]

    table1 = Table(page1_data, colWidths=[0.7 * inch, 1.2 * inch, 2.5 * inch, 1.1 * inch])
    table1.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                # Alternate row colors
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 8), (-1, 8), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 10), (-1, 10), colors.HexColor("#F5F5F5")),
            ]
        )
    )
    story.append(table1)
    story.append(Spacer(1, 15))
    story.append(Paragraph("(Continued on next page)", note_style))

    # Page 2
    story.append(PageBreak())
    story.append(Paragraph("DEPARTMENT BUDGET - FY 2024 (continued)", title_style))
    story.append(Paragraph("Information Technology Division", subtitle_style))

    page2_data = [
        ["Line", "Category", "Description", "Amount"],
        ["4001", "Services", "Consulting and contractors", "$80,000"],
        ["4002", "Services", "Maintenance contracts", "$45,000"],
        ["4003", "Services", "Internet and telecom", "$24,000"],
        ["5001", "Travel", "Conference attendance", "$12,000"],
        ["5002", "Travel", "Site visits", "$8,000"],
        ["6001", "Supplies", "Office supplies", "$5,000"],
        ["6002", "Supplies", "Hardware consumables", "$3,500"],
        ["", "", "", ""],
        ["", "", "DIVISION TOTAL:", "$1,273,500"],
    ]

    table2 = Table(page2_data, colWidths=[0.7 * inch, 1.2 * inch, 2.5 * inch, 1.1 * inch])
    table2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -2), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                # Alternate row colors
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 6), (-1, 6), colors.HexColor("#F5F5F5")),
                # Total row styling
                ("FONTNAME", (2, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (2, -1), (-1, -1), 12),
                ("LINEABOVE", (2, -1), (-1, -1), 2, colors.black),
            ]
        )
    )
    story.append(table2)

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'budget_items.pdf'}")


def create_annual_report():
    """PDF 7: Document with Sections - annual report with multiple sections."""
    doc = SimpleDocTemplate(
        str(OUTPUT_DIR / "annual_report.pdf"),
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=14,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=30,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=14,
        fontName="Helvetica-Bold",
        spaceBefore=25,
        spaceAfter=10,
        textColor=colors.HexColor("#1F4E79"),
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        spaceAfter=10,
    )

    bullet_style = ParagraphStyle(
        "Bullet",
        parent=styles["Normal"],
        fontSize=11,
        leftIndent=20,
        spaceAfter=5,
    )

    story = []

    # Title page content
    story.append(Paragraph("ANNUAL REPORT 2024", title_style))
    story.append(Paragraph("Acme Technology Solutions", subtitle_style))

    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", section_style))
    story.append(
        Paragraph(
            "This year marked a significant milestone for Acme Technology Solutions. "
            "We achieved record revenue growth of 23% year-over-year while expanding our "
            "customer base by 15%. Our commitment to innovation resulted in the launch of "
            "three new product lines, positioning us as a market leader in enterprise software.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Key achievements include successful expansion into the European market, "
            "a strategic acquisition that enhanced our AI capabilities, and recognition "
            "as one of the top 50 tech companies to work for.",
            body_style,
        )
    )

    # Financial Highlights
    story.append(Paragraph("FINANCIAL HIGHLIGHTS", section_style))

    financials = [
        ["Metric", "2024", "2023", "Change"],
        ["Revenue", "$10.5M", "$8.5M", "+23%"],
        ["Gross Profit", "$6.8M", "$5.4M", "+26%"],
        ["Operating Expenses", "$4.2M", "$3.8M", "+11%"],
        ["Net Income", "$2.6M", "$1.6M", "+63%"],
        ["R&D Investment", "$1.8M", "$1.2M", "+50%"],
    ]

    fin_table = Table(financials, colWidths=[2 * inch, 1.2 * inch, 1.2 * inch, 1 * inch])
    fin_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#F5F5F5")),
                ("BACKGROUND", (0, 4), (-1, 4), colors.HexColor("#F5F5F5")),
            ]
        )
    )
    story.append(fin_table)
    story.append(Spacer(1, 10))

    # Operational Achievements
    story.append(Paragraph("OPERATIONAL ACHIEVEMENTS", section_style))
    story.append(
        Paragraph(
            "Our operational excellence initiatives delivered significant results:", body_style
        )
    )

    achievements = [
        "Customer satisfaction score improved from 87% to 94%",
        "Average response time reduced by 40%",
        "Employee retention rate increased to 92%",
        "Successfully onboarded 45 new enterprise clients",
        "Launched mobile app with 50,000+ downloads in first quarter",
    ]

    for item in achievements:
        story.append(Paragraph(f"\u2022 {item}", bullet_style))

    # Strategic Initiatives
    story.append(Paragraph("STRATEGIC INITIATIVES", section_style))
    story.append(
        Paragraph(
            "We executed on several strategic priorities that will drive future growth. "
            "The acquisition of DataFlow Inc. for $3.2M added crucial machine learning "
            "capabilities to our platform. Our European expansion established offices in "
            "London and Berlin, with plans for Paris in Q2 2025.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Product development focused on AI-powered features, resulting in our flagship "
            "AutoAnalytics product line. Early adoption has exceeded projections by 35%.",
            body_style,
        )
    )

    # Risks and Challenges
    story.append(Paragraph("RISKS AND CHALLENGES", section_style))
    story.append(
        Paragraph(
            "While our performance was strong, we remain vigilant about market challenges. "
            "Increased competition in the enterprise segment requires continued innovation. "
            "Economic uncertainty may impact customer purchasing decisions in 2025.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "Talent acquisition remains competitive in the technology sector. We have "
            "implemented enhanced compensation packages and flexible work arrangements "
            "to attract and retain top performers.",
            body_style,
        )
    )

    # Looking Ahead
    story.append(Paragraph("LOOKING AHEAD", section_style))
    story.append(
        Paragraph(
            "For 2025, we project revenue growth of 20-25% driven by our expanded product "
            "portfolio and geographic presence. Key priorities include:",
            body_style,
        )
    )

    priorities = [
        "Launch of AutoAnalytics 2.0 with enhanced AI capabilities",
        "Expansion into Asia-Pacific market",
        "Development of strategic partnerships with major cloud providers",
        "Continued investment in R&D (target: 20% of revenue)",
    ]

    for item in priorities:
        story.append(Paragraph(f"\u2022 {item}", bullet_style))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "We thank our shareholders, customers, and employees for their continued support "
            "and look forward to another successful year.",
            body_style,
        )
    )

    doc.build(story)
    print(f"Created: {OUTPUT_DIR / 'annual_report.pdf'}")


def create_scanned_form():
    """PDF 5: Scanned Document Simulation - image-based form requiring OCR."""
    import io

    from PIL import Image as PILImage
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    output_path = OUTPUT_DIR / "scanned_form.pdf"

    # First, create a canvas with the form content
    c = canvas.Canvas(str(output_path), pagesize=letter)
    width, height = letter

    # Add some "noise" effect by using a light gray background
    c.setFillColor(colors.HexColor("#F8F6F0"))  # Slightly off-white
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # Title
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 60, "APPLICATION FORM")

    # Simulate a slightly rotated/skewed look with positioning
    y = height - 100

    # Form fields (label-value pairs)
    c.setFont("Helvetica", 11)

    fields = [
        ("Name:", "John A. Smith"),
        ("Date of Birth:", "03/15/1985"),
        ("Address:", "456 Oak Street, Apt 2B"),
        ("City, State:", "Springfield, IL 62701"),
        ("Phone:", "(555) 123-4567"),
        ("Email:", "john.smith@email.com"),
        ("Application Date:", "01/20/2024"),
        ("Reference Number:", "REF-2024-0892"),
    ]

    for label, value in fields:
        # Draw label
        c.setFont("Helvetica-Bold", 11)
        c.drawString(80, y, label)

        # Draw underline for field
        c.setStrokeColor(colors.grey)
        c.line(180, y - 2, 450, y - 2)

        # Draw value (simulating filled-in text)
        c.setFont("Helvetica", 11)
        c.drawString(185, y, value)

        y -= 35

    # Checkbox section
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(80, y, "Services Requested:")
    y -= 25

    checkboxes = [
        ("Standard Processing", True),
        ("Expedited Review", False),
        ("Document Certification", True),
        ("Additional Copies", False),
    ]

    c.setFont("Helvetica", 10)
    for label, checked in checkboxes:
        # Draw checkbox
        c.rect(80, y - 3, 12, 12, fill=0, stroke=1)
        if checked:
            # Draw checkmark
            c.setStrokeColor(colors.black)
            c.line(82, y + 2, 86, y - 1)
            c.line(86, y - 1, 90, y + 7)

        c.drawString(100, y, label)
        y -= 22

    # Amount section
    y -= 20
    c.setFont("Helvetica-Bold", 11)
    c.drawString(80, y, "Amount Enclosed:")
    c.setFont("Helvetica", 11)
    c.drawString(200, y, "$75.00")

    y -= 30
    c.setFont("Helvetica-Bold", 11)
    c.drawString(80, y, "Payment Method:")
    c.setFont("Helvetica", 11)
    c.drawString(200, y, "Check #1234")

    # Signature area
    y -= 60
    c.setFont("Helvetica-Bold", 11)
    c.drawString(80, y, "Signature:")
    c.setStrokeColor(colors.grey)
    c.line(160, y - 2, 350, y - 2)

    # Simulated signature (cursive-ish)
    c.setFont("Helvetica-Oblique", 14)
    c.drawString(170, y, "John Smith")

    y -= 30
    c.drawString(80, y, "Date:")
    c.line(130, y - 2, 250, y - 2)
    c.setFont("Helvetica", 11)
    c.drawString(135, y, "01/20/2024")

    # Add some "scan artifacts" - light spots
    c.setFillColor(colors.HexColor("#E8E8E8"))
    import random

    random.seed(42)  # Reproducible
    for _ in range(15):
        x = random.randint(50, int(width - 50))
        y_spot = random.randint(50, int(height - 50))
        r = random.randint(2, 5)
        c.circle(x, y_spot, r, fill=1, stroke=0)

    # Footer
    c.setFillColor(colors.grey)
    c.setFont("Helvetica", 8)
    c.drawString(80, 50, "Form Rev. 2024-01 | Page 1 of 1")

    c.save()
    print(f"Created: {output_path}")


def main():
    """Create all cookbook sample PDFs."""
    print("Creating cookbook sample PDFs...\n")

    create_quarterly_sales()
    create_vendor_invoice()
    create_facility_inspections()
    create_incident_log()
    create_scanned_form()
    create_budget_items()
    create_annual_report()

    print(f"\nAll PDFs created in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
