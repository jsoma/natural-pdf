# natural_pdf/widgets/viewer.py

import logging

from natural_pdf.utils.visualization import render_plain_page

logger = logging.getLogger(__name__)

# Initialize flag and module/class variables to None
_IPYWIDGETS_AVAILABLE = False
widgets = None
InteractiveViewerWidget = None

try:
    # Attempt to import the core optional dependency
    import ipywidgets as widgets_imported

    widgets = widgets_imported  # Assign to the global name if import succeeds
    _IPYWIDGETS_AVAILABLE = True
    logger.debug("Successfully imported ipywidgets. Defining viewer widgets.")

    # --- Dependencies needed ONLY if ipywidgets is available ---
    import base64
    import json
    import uuid
    from io import BytesIO

    from IPython.display import HTML, Javascript, display
    from PIL import Image

    # --- Define Widget Class ---
    class InteractiveViewerWidget(widgets.DOMWidget):
        def __init__(self, pdf_data=None, **kwargs):
            """
            Create an interactive PDF viewer widget.

            Args:
                pdf_data (dict, optional): Dictionary containing 'page_image', 'elements', etc.
                **kwargs: Additional parameters including image_uri, elements, etc.
            """
            super().__init__()

            # Support both pdf_data dict and individual kwargs
            if pdf_data:
                self.pdf_data = pdf_data
                # Ensure backward compatibility - if image_uri exists but page_image doesn't
                if "image_uri" in pdf_data and "page_image" not in pdf_data:
                    self.pdf_data["page_image"] = pdf_data["image_uri"]
            else:
                # Check for image_uri in kwargs
                image_source = kwargs.get("image_uri", "")

                self.pdf_data = {"page_image": image_source, "elements": kwargs.get("elements", [])}

            # Log for debugging
            logger.debug(f"InteractiveViewerWidget initialized with widget_id={id(self)}")
            logger.debug(
                f"Image source provided: {self.pdf_data.get('page_image', 'None')[:30]}..."
            )
            logger.debug(f"Number of elements: {len(self.pdf_data.get('elements', []))}")

            self.widget_id = f"pdf-viewer-{str(uuid.uuid4())[:8]}"
            self._generate_html()

        def _generate_html(self):
            """Generate the HTML for the PDF viewer"""
            # Extract data - Coordinates in self.pdf_data['elements'] are already scaled
            page_image = self.pdf_data.get("page_image", "")
            elements = self.pdf_data.get("elements", [])

            logger.debug(
                f"Generating HTML with image: {page_image[:30]}... and {len(elements)} elements (using scaled coords)"
            )

            # Create the container div
            container_html = f"""
            <div id="{self.widget_id}" class="pdf-viewer" style="position: relative; font-family: Arial, sans-serif;">
                <div class="toolbar" style="margin-bottom: 10px; padding: 5px; background-color: #f0f0f0; border-radius: 4px;">
                    <button id="{self.widget_id}-zoom-in" style="margin-right: 5px;">Zoom In (+)</button>
                    <button id="{self.widget_id}-zoom-out" style="margin-right: 5px;">Zoom Out (-)</button>
                    <button id="{self.widget_id}-reset-zoom" style="margin-right: 5px;">Reset</button>
                </div>
                <div style="display: flex; flex-direction: row;">
                    <div class="pdf-outer-container" style="position: relative; overflow: hidden; border: 1px solid #ccc; flex-grow: 1;"> 
                        <div id="{self.widget_id}-zoom-pan-container" class="zoom-pan-container" style="position: relative; width: fit-content; height: fit-content; transform-origin: top left; cursor: grab;">
                        <!-- The image is rendered at scale, so its dimensions match scaled coordinates -->
                            <img src="{page_image}" style="display: block; max-width: none; height: auto;" /> 
                        <div id="{self.widget_id}-elements-layer" class="elements-layer" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
            """

            # Add SVG overlay layer
            container_html += f"""
                        </div>
                        <div id="{self.widget_id}-svg-layer" class="svg-layer" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;">
                            <!-- SVG viewport should match the scaled image size -->
                            <svg width="100%" height="100%">
            """

            # Add elements and SVG boxes using the SCALED coordinates
            for i, element in enumerate(elements):
                element_type = element.get("type", "unknown")
                # Use the already scaled coordinates
                x0 = element.get("x0", 0)
                y0 = element.get("y0", 0)
                x1 = element.get("x1", 0)
                y1 = element.get("y1", 0)

                # Calculate width and height from scaled coords
                width = x1 - x0
                height = y1 - y0

                # Create the element div with the right styling based on type
                # Use scaled coordinates for positioning and dimensions
                element_style = "position: absolute; pointer-events: auto; cursor: pointer; "
                element_style += (
                    f"left: {x0}px; top: {y0}px; width: {width}px; height: {height}px; "
                )

                # Different styling for different element types
                if element_type == "text":
                    element_style += (
                        "background-color: rgba(255, 255, 0, 0.3); border: 1px dashed transparent; "
                    )
                elif element_type == "image":
                    element_style += (
                        "background-color: rgba(0, 128, 255, 0.3); border: 1px dashed transparent; "
                    )
                elif element_type == "figure":
                    element_style += (
                        "background-color: rgba(255, 0, 255, 0.3); border: 1px dashed transparent; "
                    )
                elif element_type == "table":
                    element_style += (
                        "background-color: rgba(0, 255, 0, 0.3); border: 1px dashed transparent; "
                    )
                else:
                    element_style += "background-color: rgba(200, 200, 200, 0.3); border: 1px dashed transparent; "

                # Add element div
                container_html += f"""
                            <div class="pdf-element" data-element-id="{i}" style="{element_style}"></div>
                """

                # Add SVG rectangle using scaled coordinates and dimensions
                container_html += f"""
                            <rect data-element-id="{i}" x="{x0}" y="{y0}" width="{width}" height="{height}" 
                                  fill="none" stroke="rgba(255, 165, 0, 0.85)" stroke-width="1.5" />
                """

            # Close SVG and container divs
            container_html += f"""
                            </svg>
                        </div>
                    </div>
                </div>
                
                <div id="{self.widget_id}-info-panel" class="info-panel" style="display: block; margin-left: 20px; padding: 10px; width: 300px; max-height: 80vh; overflow-y: auto; border: 1px solid #eee; background-color: #f9f9f9;">
                    <h4 style="margin-top: 0; margin-bottom: 5px; border-bottom: 1px solid #ccc; padding-bottom: 5px;">Element Info</h4>
                    <pre id="{self.widget_id}-element-data" style="white-space: pre-wrap; word-break: break-all; font-size: 0.9em;"></pre>
                </div>
                
            </div>
            """

            # Display the HTML
            display(HTML(container_html))

            # Generate JavaScript to add interactivity
            self._add_javascript()

        def _add_javascript(self):
            """Add JavaScript to make the viewer interactive"""
            # Create JavaScript for element selection and SVG highlighting
            js_code = """
            (function() {
                // Store widget ID in a variable to avoid issues with string templates
                const widgetId = "%s";
                
                // Initialize PDF viewer registry if it doesn't exist
                if (!window.pdfViewerRegistry) {
                    window.pdfViewerRegistry = {};
                }
                
                // Store PDF data for this widget
                window.pdfViewerRegistry[widgetId] = {
                    initialData: %s,
                    selectedElement: null,
                    scale: 1.0,         // Initial zoom scale
                    translateX: 0,    // Initial pan X
                    translateY: 0,    // Initial pan Y
                    isDragging: false, // Flag for panning
                    startX: 0,          // Drag start X
                    startY: 0,          // Drag start Y
                    startTranslateX: 0, // Translate X at drag start
                    startTranslateY: 0, // Translate Y at drag start
                    justDragged: false // Flag to differentiate click from drag completion
                };
                
                // Get references to elements
                const viewerData = window.pdfViewerRegistry[widgetId];
                const outerContainer = document.querySelector(`#${widgetId} .pdf-outer-container`);
                const zoomPanContainer = document.getElementById(`${widgetId}-zoom-pan-container`);
                const elements = zoomPanContainer.querySelectorAll(".pdf-element");
                const zoomInButton = document.getElementById(`${widgetId}-zoom-in`);
                const zoomOutButton = document.getElementById(`${widgetId}-zoom-out`);
                const resetButton = document.getElementById(`${widgetId}-reset-zoom`);
                
                // --- Helper function to apply transform --- 
                function applyTransform() {
                    zoomPanContainer.style.transform = `translate(${viewerData.translateX}px, ${viewerData.translateY}px) scale(${viewerData.scale})`;
                }
                
                // --- Zooming Logic --- 
                function handleZoom(event) {
                    event.preventDefault(); // Prevent default scroll
                    
                    const zoomIntensity = 0.1;
                    const wheelDelta = event.deltaY < 0 ? 1 : -1; // +1 for zoom in, -1 for zoom out
                    const zoomFactor = Math.exp(wheelDelta * zoomIntensity);
                    const newScale = Math.max(0.5, Math.min(5, viewerData.scale * zoomFactor)); // Clamp scale
                    
                    // Calculate mouse position relative to the outer container
                    const rect = outerContainer.getBoundingClientRect();
                    const mouseX = event.clientX - rect.left;
                    const mouseY = event.clientY - rect.top;
                    
                    // Calculate the point in the content that the mouse is pointing to
                    const pointX = (mouseX - viewerData.translateX) / viewerData.scale;
                    const pointY = (mouseY - viewerData.translateY) / viewerData.scale;
                    
                    // Update scale
                    viewerData.scale = newScale;
                    
                    // Calculate new translation to keep the pointed-at location fixed
                    viewerData.translateX = mouseX - pointX * viewerData.scale;
                    viewerData.translateY = mouseY - pointY * viewerData.scale;
                    
                    applyTransform();
                }
                
                outerContainer.addEventListener('wheel', handleZoom);
                
                // --- Panning Logic --- 
                const dragThreshold = 5; // Pixels to move before drag starts

                function handleMouseDown(event) {
                    // Prevent default only if needed (e.g., text selection on image)
                    if (event.target.tagName !== 'BUTTON') {
                        event.preventDefault();
                    }
                    
                    viewerData.isDragging = true;
                    viewerData.startX = event.clientX;
                    viewerData.startY = event.clientY;
                    viewerData.startTranslateX = viewerData.translateX;
                    viewerData.startTranslateY = viewerData.translateY;
                    viewerData.justDragged = false; // Reset drag flag
                    zoomPanContainer.style.cursor = 'grabbing';
                }

                function handleMouseMove(event) {
                    if (!viewerData.isDragging) return;

                    const dx = event.clientX - viewerData.startX;
                    const dy = event.clientY - viewerData.startY;
                    
                    // If we've moved past the threshold, it's a drag
                    if (Math.abs(dx) > dragThreshold || Math.abs(dy) > dragThreshold) {
                         viewerData.justDragged = true;
                    }

                    viewerData.translateX = viewerData.startTranslateX + dx;
                    viewerData.translateY = viewerData.startTranslateY + dy;
                    applyTransform();
                }

                function handleMouseUp() {
                    viewerData.isDragging = false;
                    zoomPanContainer.style.cursor = 'grab';
                }
                
                zoomPanContainer.addEventListener('mousedown', handleMouseDown);
                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
                
                // --- Button Controls ---
                zoomInButton.addEventListener('click', () => {
                     viewerData.scale = Math.min(5, viewerData.scale * 1.2);
                     applyTransform();
                });
                
                 zoomOutButton.addEventListener('click', () => {
                     viewerData.scale = Math.max(0.5, viewerData.scale / 1.2);
                     applyTransform();
                });
                
                resetButton.addEventListener('click', () => {
                    viewerData.scale = 1.0;
                    viewerData.translateX = 0;
                    viewerData.translateY = 0;
                    applyTransform();
                });

                // --- Element Interaction ---
                function highlightElement(elementId) {
                    // Remove previous highlights on SVG rects
                    const allRects = zoomPanContainer.querySelectorAll('svg rect');
                    allRects.forEach(rect => {
                        rect.style.stroke = 'rgba(255, 165, 0, 0.85)'; 
                        rect.style.strokeWidth = '1.5';
                    });
                    
                    // Highlight the new one
                    const targetRect = zoomPanContainer.querySelector(`svg rect[data-element-id='${elementId}']`);
                    if (targetRect) {
                        targetRect.style.stroke = 'red';
                        targetRect.style.strokeWidth = '3';
                    }
                }
                
                function updateInfoPanel(element) {
                    const infoPanel = document.getElementById(`${widgetId}-element-data`);
                    if (infoPanel) {
                         // Pretty print the JSON
                         let displayData = {};
                         for (const [key, value] of Object.entries(element)) {
                             if (key !== 'bbox') { // Exclude raw bbox
                                 if (typeof value === 'number') {
                                     displayData[key] = parseFloat(value.toFixed(2));
                                 } else {
                                     displayData[key] = value;
                                 }
                             }
                         }
                         infoPanel.textContent = JSON.stringify(displayData, null, 2);
                    }
                }

                elements.forEach(el => {
                    el.addEventListener('click', function(event) {
                        if (viewerData.justDragged) {
                            // If a drag just ended, prevent the click action
                            viewerData.justDragged = false;
                            return;
                        }
                        
                        event.stopPropagation(); // Stop click from propagating to the container
                        const elementId = this.getAttribute('data-element-id');
                        const elementData = viewerData.initialData.elements[elementId];
                        
                        console.log('Clicked element:', elementData);
                        viewerData.selectedElement = elementData;

                        // Update UI
                        updateInfoPanel(elementData);
                        highlightElement(elementId);
                        
                        // Example of sending data back to Python kernel
                        if (window.IPython && window.IPython.notebook && window.IPython.notebook.kernel) {
                            const command = `import json; from natural_pdf.widgets.viewer import InteractiveViewerWidget; InteractiveViewerWidget._handle_element_click(json.loads('${JSON.stringify(elementData)}'))`;
                            console.log("Executing command:", command);
                           // window.IPython.notebook.kernel.execute(command);
                        }
                    });
                });
            })();
            """ % (
                self.widget_id,
                json.dumps(self.pdf_data),
            )
            # Display the JavaScript
            display(Javascript(js_code))

        def _get_element_json(self):
            """Returns the elements as a JSON string."""
            # We don't need to do anything special here as the coords are already scaled
            return json.dumps(self.pdf_data.get("elements", []))

        def _repr_html_(self):
            """Called by Jupyter to display the widget."""
            # The __init__ method already calls display(), so nothing more is needed here
            return None

        @classmethod
        def from_page(cls, page, on_element_click=None, include_attributes=None):
            """
            Factory method to create a viewer from a Page object.

            Args:
                page (Page): The Page object to display.
                on_element_click (callable, optional): Callback function when an element is clicked.
                include_attributes (list, optional): List of element attributes to include.

            Returns:
                An instance of InteractiveViewerWidget.
            """
            if not _IPYWIDGETS_AVAILABLE:
                logger.warning(
                    "Optional dependency 'ipywidgets' not found. Cannot create interactive viewer."
                )
                return None

            try:
                # --- This logic is restored from the original SimpleInteractiveViewerWidget ---

                resolution = 150  # Define resolution to calculate scale
                scale = resolution / 72.0  # PDF standard DPI is 72

                # Get the page image, rendered at the higher resolution
                img = render_plain_page(page, resolution=resolution)

                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_uri = f"data:image/png;base64,{img_str}"

                # Convert elements to dict format
                elements = []
                # Use page.elements directly if available, otherwise fallback to find_all
                page_elements = getattr(page, "elements", page.find_all("*"))

                # Filter out 'char' elements which are too noisy for the viewer
                filtered_page_elements = [
                    el for el in page_elements if str(getattr(el, "type", "")).lower() != "char"
                ]

                # Define a list of common/useful attributes to check for
                default_attributes_to_get = [
                    "text",
                    "fontname",
                    "size",
                    "bold",
                    "italic",
                    "color",
                    "linewidth",
                    "is_horizontal",
                    "is_vertical",
                    "source",
                    "confidence",
                    "label",
                    "model",
                    "upright",
                    "direction",
                ]

                for i, element in enumerate(filtered_page_elements):
                    elem_dict = {
                        "id": i,
                        "type": element.type,
                        # Apply scaling to all coordinates and dimensions
                        "x0": element.x0 * scale,
                        "y0": element.top * scale,
                        "x1": element.x1 * scale,
                        "y1": element.bottom * scale,
                        "width": element.width * scale,
                        "height": element.height * scale,
                    }

                    # Get Default and User-Requested Attributes
                    attributes_found = set()
                    all_attrs_to_check = default_attributes_to_get + (include_attributes or [])

                    for attr_name in all_attrs_to_check:
                        if attr_name not in attributes_found and hasattr(element, attr_name):
                            try:
                                value = getattr(element, attr_name)
                                # Ensure value is JSON serializable
                                if not isinstance(
                                    value, (str, int, float, bool, list, dict, type(None))
                                ):
                                    value = str(value)
                                elem_dict[attr_name] = value
                                attributes_found.add(attr_name)
                            except Exception as e:
                                logger.warning(
                                    f"Could not get attribute '{attr_name}' for element {i}: {e}"
                                )

                    # Round float values for cleaner display
                    for key, val in elem_dict.items():
                        if isinstance(val, float):
                            elem_dict[key] = round(val, 2)

                    elements.append(elem_dict)

                viewer_data = {"page_image": image_uri, "elements": elements}
                # --- End of restored logic ---

                # Set the callback if provided
                if on_element_click:
                    cls._on_element_click_callback = on_element_click

                return cls(pdf_data=viewer_data)

            except Exception as e:
                logger.error(f"Failed to create viewer from page: {e}", exc_info=True)
                return None

        # Static callback storage and handler
        _on_element_click_callback = None

        @staticmethod
        def _handle_element_click(element_data):
            """Static method to handle element click events from JavaScript."""
            if InteractiveViewerWidget._on_element_click_callback:
                try:
                    InteractiveViewerWidget._on_element_click_callback(element_data)
                except Exception as e:
                    logger.error(f"Error in element click callback: {e}", exc_info=True)

except ImportError:
    # This block runs if 'ipywidgets' is not installed
    logger.info(
        "Optional dependency 'ipywidgets' not found. Interactive viewer widgets will not be defined."
    )
    # Ensure flag is False if the import fails for any reason
    _IPYWIDGETS_AVAILABLE = False
except Exception as e:
    # Catch other potential errors during widget definition
    logger.error(f"An unexpected error occurred while defining viewer widgets: {e}", exc_info=True)
    _IPYWIDGETS_AVAILABLE = False  # Explicitly set flag to False here too
