"""Interactive checkbox annotation widget for Jupyter notebooks.

Allows users to draw rectangles on a rendered page image to mark
checkbox locations. Results are returned as Region objects with
proper PDF coordinates.
"""

import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy ipywidgets check
_IPYWIDGETS_AVAILABLE = False
widgets: Any = None

try:
    import ipywidgets as _widgets

    widgets = _widgets
    _IPYWIDGETS_AVAILABLE = True
except ImportError:
    pass


class CheckboxAnnotator:
    """Interactive tool for drawing checkbox boxes on a page image.

    Usage:
        annotator = CheckboxAnnotator(page)
        annotator.show()
        # User draws boxes interactively...
        regions = annotator.get_regions()
    """

    def __init__(self, page: Any, resolution: int = 150):
        """Initialize the annotator.

        Args:
            page: Page object to annotate.
            resolution: DPI for rendering the page image.
        """
        if not _IPYWIDGETS_AVAILABLE:
            raise ImportError(
                "ipywidgets is required for interactive annotation. "
                "Install with: pip install ipywidgets"
            )

        self._page = page
        self._resolution = resolution
        self._drawn_boxes: List[Tuple[float, float, float, float]] = []
        self._widget = None

    def show(self):
        """Display interactive annotation widget."""
        import base64
        import json
        import uuid
        from io import BytesIO

        from IPython.display import HTML, display

        # Render page image
        image = self._page.render(resolution=self._resolution)
        if image is None:
            print("Failed to render page image")
            return

        img_w, img_h = image.size
        scale_x = self._page.width / img_w
        scale_y = self._page.height / img_h

        # Convert image to base64
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        widget_id = f"cb-annotator-{uuid.uuid4().hex[:8]}"

        # Store state in a text widget for communication
        self._state_widget = widgets.Textarea(
            value="[]",
            layout=widgets.Layout(display="none"),
        )
        self._state_widget.observe(self._on_state_change, names="value")

        # Control buttons
        btn_undo = widgets.Button(description="Undo", button_style="warning")
        btn_clear = widgets.Button(description="Clear", button_style="danger")
        btn_done = widgets.Button(description="Done", button_style="success")
        self._status = widgets.Label(value="Draw rectangles on checkboxes. Click and drag.")

        def on_undo(b):
            if self._drawn_boxes:
                self._drawn_boxes.pop()
                self._status.value = f"{len(self._drawn_boxes)} boxes drawn. Undo successful."

        def on_clear(b):
            self._drawn_boxes.clear()
            self._status.value = "All boxes cleared."

        def on_done(b):
            self._status.value = (
                f"Done! {len(self._drawn_boxes)} boxes finalized. Call get_regions() to retrieve."
            )

        btn_undo.on_click(on_undo)
        btn_clear.on_click(on_clear)
        btn_done.on_click(on_done)

        controls = widgets.HBox([btn_undo, btn_clear, btn_done])

        # HTML canvas with drawing support
        canvas_html = f"""
        <div id="{widget_id}" style="position:relative; display:inline-block; cursor:crosshair;">
            <img src="data:image/png;base64,{img_b64}"
                 style="max-width:100%; display:block;"
                 id="{widget_id}-img" />
            <canvas id="{widget_id}-canvas"
                    width="{img_w}" height="{img_h}"
                    style="position:absolute; top:0; left:0; width:100%; height:100%;"></canvas>
        </div>
        <script>
        (function() {{
            var container = document.getElementById('{widget_id}');
            var canvas = document.getElementById('{widget_id}-canvas');
            var ctx = canvas.getContext('2d');
            var boxes = [];
            var drawing = false;
            var startX, startY, curX, curY;

            function getPos(e) {{
                var rect = canvas.getBoundingClientRect();
                var scaleX = canvas.width / rect.width;
                var scaleY = canvas.height / rect.height;
                return {{
                    x: (e.clientX - rect.left) * scaleX,
                    y: (e.clientY - rect.top) * scaleY
                }};
            }}

            function redraw() {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.strokeStyle = '#FF0000';
                ctx.lineWidth = 2;
                for (var i = 0; i < boxes.length; i++) {{
                    var b = boxes[i];
                    ctx.strokeRect(b[0], b[1], b[2]-b[0], b[3]-b[1]);
                }}
            }}

            canvas.addEventListener('mousedown', function(e) {{
                var pos = getPos(e);
                startX = pos.x;
                startY = pos.y;
                drawing = true;
            }});

            canvas.addEventListener('mousemove', function(e) {{
                if (!drawing) return;
                var pos = getPos(e);
                curX = pos.x;
                curY = pos.y;
                redraw();
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 2;
                ctx.setLineDash([5, 3]);
                ctx.strokeRect(startX, startY, curX-startX, curY-startY);
                ctx.setLineDash([]);
            }});

            canvas.addEventListener('mouseup', function(e) {{
                if (!drawing) return;
                drawing = false;
                var pos = getPos(e);
                var x0 = Math.min(startX, pos.x);
                var y0 = Math.min(startY, pos.y);
                var x1 = Math.max(startX, pos.x);
                var y1 = Math.max(startY, pos.y);
                if ((x1-x0) > 3 && (y1-y0) > 3) {{
                    // Convert to PDF coords
                    var pdf_x0 = x0 * {scale_x};
                    var pdf_y0 = y0 * {scale_y};
                    var pdf_x1 = x1 * {scale_x};
                    var pdf_y1 = y1 * {scale_y};
                    boxes.push([pdf_x0, pdf_y0, pdf_x1, pdf_y1]);
                }}
                redraw();
            }});
        }})();
        </script>
        """

        html_widget = widgets.HTML(value=canvas_html)
        display(self._status)
        display(html_widget)
        display(controls)

        self._img_w = img_w
        self._img_h = img_h
        self._scale_x = scale_x
        self._scale_y = scale_y

    def add_box(self, x0: float, y0: float, x1: float, y1: float):
        """Programmatically add a box in PDF coordinates."""
        self._drawn_boxes.append((x0, y0, x1, y1))

    def _on_state_change(self, change):
        """Handle state updates from JavaScript."""
        import json

        try:
            boxes = json.loads(change["new"])
            self._drawn_boxes = [tuple(b) for b in boxes if len(b) == 4]
        except (json.JSONDecodeError, TypeError):
            pass

    def get_regions(
        self,
        classify: bool = True,
        classify_with: Optional[Any] = None,
    ) -> List[Any]:
        """Convert drawn boxes to checkbox Region objects.

        Args:
            classify: Whether to run pixel classification.
            classify_with: Optional Judge instance.

        Returns:
            List of Region objects.
        """
        from natural_pdf.analyzers.checkbox.classifier import CheckboxClassifier

        regions = []
        for box in self._drawn_boxes:
            x0, y0, x1, y1 = box
            region = self._page.create_region(x0, y0, x1, y1)
            region.region_type = "checkbox"
            region.normalized_type = "checkbox"
            region.is_checked = None
            region.checkbox_state = "unknown"
            region.confidence = 1.0  # Manual annotation
            region.model = "manual"
            region.source = "manual"

            region.analyses["checkbox"] = {
                "is_checked": None,
                "state": "unknown",
                "confidence": 1.0,
                "model": "manual",
                "engine": "manual",
            }

            regions.append(region)
            self._page.add_region(region, source="manual")

        # Classify if requested
        if classify and regions:
            CheckboxClassifier.classify_regions(
                regions, self._page, judge=classify_with, classify=True
            )

        return regions
