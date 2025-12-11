from natural_pdf.describe.elements import describe_line_elements, describe_rect_elements


class _Rect:
    def __init__(self, *, width=10, height=10, stroke=None, fill=None, stroke_width=1):
        self.width = width
        self.height = height
        self.stroke = stroke
        self.fill = fill
        self.stroke_width = stroke_width


class _Line:
    def __init__(self, *, color=None, width=1, x0=0, y0=0, x1=5, y1=0):
        self.color = color
        self.width = width
        self.x0 = x0
        self.top = y0
        self.x1 = x1
        self.bottom = y1


def test_rect_describe_reports_hex_colors():
    elements = [
        _Rect(stroke=(0, 0, 0)),
        _Rect(fill=(1, 0, 0)),
        _Rect(stroke=(0.2, 0.4, 0.6)),
    ]
    summary = describe_rect_elements(elements)
    colors = summary["styles"]["colors"]
    assert colors["#000000"] == 1
    assert colors["#FF0000"] == 1
    assert colors["#336699"] == 1


def test_line_describe_reports_hex_colors():
    elements = [
        _Line(color=(0, 0, 0)),
        _Line(color=(128, 64, 32)),
    ]
    summary = describe_line_elements(elements)
    colors = summary["colors"]
    assert colors["#000000"] == 1
    assert colors["#804020"] == 1
