# natural_pdf/widgets/viewer.py

import ipywidgets as widgets
from traitlets import Unicode, List, Dict, observe
import os
import logging # Add logging
import json
from IPython.display import display, HTML, Javascript
import uuid
from PIL import Image

logger = logging.getLogger(__name__)

# --- Read JS code from file --- #
_MODULE_DIR = os.path.dirname(__file__)
_FRONTEND_JS_PATH = os.path.join(_MODULE_DIR, 'frontend', 'viewer.js')

try:
    with open(_FRONTEND_JS_PATH, 'r', encoding='utf-8') as f:
        _FRONTEND_JS_CODE = f.read()
    logger.debug(f"Successfully read frontend JS from: {_FRONTEND_JS_PATH}")
except FileNotFoundError:
     logger.error(f"Frontend JS file not found at {_FRONTEND_JS_PATH}. Widget will likely fail.")
     _FRONTEND_JS_CODE = "console.error('Frontend JS file not found! Widget cannot load.');"
except Exception as e:
    logger.error(f"Error reading frontend JS file {_FRONTEND_JS_PATH}: {e}")
    _FRONTEND_JS_CODE = f"console.error('Error reading frontend JS file: {e}');"

class SimpleInteractiveViewerWidget(widgets.DOMWidget):
    def __init__(self, pdf_data=None, **kwargs):
        """
        Create a simple interactive PDF viewer widget.
        
        Args:
            pdf_data (dict, optional): Dictionary containing 'page_image', 'elements', etc.
            **kwargs: Additional parameters including image_uri, elements, etc.
        """
        super().__init__()
        
        # Support both pdf_data dict and individual kwargs
        if pdf_data:
            self.pdf_data = pdf_data
            # Ensure backward compatibility - if image_uri exists but page_image doesn't
            if 'image_uri' in pdf_data and 'page_image' not in pdf_data:
                self.pdf_data['page_image'] = pdf_data['image_uri']
        else:
            # Check for image_uri in kwargs
            image_source = kwargs.get('image_uri', '')
            
            self.pdf_data = {
                'page_image': image_source,
                'elements': kwargs.get('elements', [])
            }
        
        # Log for debugging
        logger.debug(f"SimpleInteractiveViewerWidget initialized with widget_id={id(self)}")
        logger.debug(f"Image source provided: {self.pdf_data.get('page_image', 'None')[:30]}...")
        logger.debug(f"Number of elements: {len(self.pdf_data.get('elements', []))}")
            
        self.widget_id = f"pdf-viewer-{str(uuid.uuid4())[:8]}"
        self._generate_html()
        
    def _generate_html(self):
        """Generate the HTML for the PDF viewer"""
        # Extract data - Coordinates in self.pdf_data['elements'] are already scaled
        page_image = self.pdf_data.get('page_image', '')
        elements = self.pdf_data.get('elements', [])
        
        logger.debug(f"Generating HTML with image: {page_image[:30]}... and {len(elements)} elements (using scaled coords)")
        
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
            element_type = element.get('type', 'unknown')
            # Use the already scaled coordinates
            x0 = element.get('x0', 0)
            y0 = element.get('y0', 0)
            x1 = element.get('x1', 0)
            y1 = element.get('y1', 0)
            
            # Calculate width and height from scaled coords
            width = x1 - x0
            height = y1 - y0
            
            # Create the element div with the right styling based on type
            # Use scaled coordinates for positioning and dimensions
            element_style = "position: absolute; pointer-events: auto; cursor: pointer; "
            element_style += f"left: {x0}px; top: {y0}px; width: {width}px; height: {height}px; "
            
            # Different styling for different element types
            if element_type == 'text':
                element_style += "background-color: rgba(255, 255, 0, 0.3); border: 1px dashed transparent; "
            elif element_type == 'image':
                element_style += "background-color: rgba(0, 128, 255, 0.3); border: 1px dashed transparent; "
            elif element_type == 'figure':
                element_style += "background-color: rgba(255, 0, 255, 0.3); border: 1px dashed transparent; "
            elif element_type == 'table':
                element_style += "background-color: rgba(0, 255, 0, 0.3); border: 1px dashed transparent; "
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
                 if (event.target.tagName === 'IMG') {
                    event.preventDefault();
                 }
                // Allow mousedown events on elements to proceed for potential clicks
                // Record start position for potential drag
                viewerData.startX = event.clientX;
                viewerData.startY = event.clientY;
                // Store initial translate values to calculate relative movement
                viewerData.startTranslateX = viewerData.translateX;
                viewerData.startTranslateY = viewerData.translateY;
                // Don't set isDragging = true yet
                // Don't change pointerEvents yet
            }
            
            function handleMouseMove(event) {
                // Check if mouse button is actually down (browser inconsistencies)
                if (event.buttons !== 1) { 
                     if (viewerData.isDragging) {
                         // Force drag end if button is released unexpectedly
                         handleMouseUp(event); 
                     }
                     return; 
                 }

                const currentX = event.clientX;
                const currentY = event.clientY;
                const deltaX = currentX - viewerData.startX;
                const deltaY = currentY - viewerData.startY;

                // If not already dragging, check if threshold is exceeded
                if (!viewerData.isDragging) {
                    const movedDistance = Math.hypot(deltaX, deltaY);
                    if (movedDistance > dragThreshold) {
                        viewerData.isDragging = true;
                        zoomPanContainer.style.cursor = 'grabbing';
                        // Now disable pointer events on elements since a drag has started
                        elements.forEach(el => el.style.pointerEvents = 'none');
                    }
                }

                // If dragging, update transform
                if (viewerData.isDragging) {
                     // Prevent text selection during drag
                     event.preventDefault(); 
                    viewerData.translateX = viewerData.startTranslateX + deltaX;
                    viewerData.translateY = viewerData.startTranslateY + deltaY;
                    applyTransform();
                }
            }
            
            function handleMouseUp(event) {
                const wasDragging = viewerData.isDragging;
                
                 // Always reset cursor on mouse up
                zoomPanContainer.style.cursor = 'grab';

                if (wasDragging) {
                     viewerData.isDragging = false;
                    // Restore pointer events now that drag is finished
                    elements.forEach(el => el.style.pointerEvents = 'auto');
                    
                    // Set flag to indicate a drag just finished
                    viewerData.justDragged = true;
                    // Reset the flag after a minimal delay, allowing the click event to be ignored
                    setTimeout(() => { viewerData.justDragged = false; }, 0);

                    // IMPORTANT: Prevent this mouseup from triggering other default actions
                    event.preventDefault(); 
                    // Stop propagation might not be needed here if the click listener checks justDragged
                    // event.stopPropagation(); 
                 } else {
                     // If it wasn't a drag, do nothing here. 
                     // The browser should naturally fire a 'click' event on the target element 
                     // which will be handled by the element's specific click listener 
                     // or the outerContainer's listener if it was on the background.
                 }
            }
            
            // Mousedown starts the *potential* for a drag
            // Attach to outer container to catch drags starting anywhere inside
            outerContainer.addEventListener('mousedown', handleMouseDown);
            
            // Mousemove determines if it's *actually* a drag and updates position
            // Attach to window or document for smoother dragging even if mouse leaves outerContainer
            // Using outerContainer for now, might need adjustment if dragging feels jerky near edges
            outerContainer.addEventListener('mousemove', handleMouseMove); 
            
            // Mouseup ends the drag *or* allows a click to proceed
             // Attach to window or document to ensure drag ends even if mouse released outside
             // Using outerContainer for now
            outerContainer.addEventListener('mouseup', handleMouseUp);
            
            // Stop dragging if mouse leaves the outer container entirely (optional but good practice)
            outerContainer.addEventListener('mouseleave', (event) => {
                 // Only act if the primary mouse button is NOT pressed anymore when leaving
                 if (viewerData.isDragging && event.buttons !== 1) { 
                     handleMouseUp(event); 
                 }
             }); 
            
            // --- Button Listeners --- 
            zoomInButton.addEventListener('click', () => {
                const centerRect = outerContainer.getBoundingClientRect();
                const centerX = centerRect.width / 2;
                const centerY = centerRect.height / 2;
                const zoomFactor = 1.2;
                const newScale = Math.min(5, viewerData.scale * zoomFactor);
                const pointX = (centerX - viewerData.translateX) / viewerData.scale;
                const pointY = (centerY - viewerData.translateY) / viewerData.scale;
                viewerData.scale = newScale;
                viewerData.translateX = centerX - pointX * viewerData.scale;
                viewerData.translateY = centerY - pointY * viewerData.scale;
                applyTransform();
            });
            
            zoomOutButton.addEventListener('click', () => {
                 const centerRect = outerContainer.getBoundingClientRect();
                const centerX = centerRect.width / 2;
                const centerY = centerRect.height / 2;
                const zoomFactor = 1 / 1.2;
                const newScale = Math.max(0.5, viewerData.scale * zoomFactor);
                const pointX = (centerX - viewerData.translateX) / viewerData.scale;
                const pointY = (centerY - viewerData.translateY) / viewerData.scale;
                viewerData.scale = newScale;
                viewerData.translateX = centerX - pointX * viewerData.scale;
                viewerData.translateY = centerY - pointY * viewerData.scale;
                applyTransform();
            });
            
            resetButton.addEventListener('click', () => {
                viewerData.scale = 1.0;
                viewerData.translateX = 0;
                viewerData.translateY = 0;
                applyTransform();
                // Also reset selection on zoom reset
                if (viewerData.selectedElement !== null) {
                    resetElementStyle(viewerData.selectedElement);
                    viewerData.selectedElement = null;
                    // Optionally clear info panel
                    // const elementData = document.getElementById(widgetId + "-element-data");
                    // if (elementData) elementData.textContent = '';
                }
            });
            
            // --- Helper function to reset element style ---
            function resetElementStyle(elementIdx) {
                const el = zoomPanContainer.querySelector(`.pdf-element[data-element-id='${elementIdx}']`);
                const svgRect = document.querySelector(`#${widgetId} .svg-layer svg rect[data-element-id='${elementIdx}']`);
                if (!el) return;

                const viewer = window.pdfViewerRegistry[widgetId];
                const eType = viewer.initialData.elements[elementIdx].type || 'unknown';

                if (eType === 'text') {
                    el.style.backgroundColor = "rgba(255, 255, 0, 0.3)";
                } else if (eType === 'image') {
                    el.style.backgroundColor = "rgba(0, 128, 255, 0.3)";
                } else if (eType === 'figure') {
                    el.style.backgroundColor = "rgba(255, 0, 255, 0.3)";
                } else if (eType === 'table') {
                    el.style.backgroundColor = "rgba(0, 255, 0, 0.3)";
                } else {
                    el.style.backgroundColor = "rgba(200, 200, 200, 0.3)";
                }
                el.style.border = "1px dashed transparent";

                if (svgRect) {
                    svgRect.setAttribute("stroke", "rgba(255, 165, 0, 0.85)");
                    svgRect.setAttribute("stroke-width", "1.5");
                }
            }

            // --- Helper function to set element style (selected/hover) ---
            function setElementHighlightStyle(elementIdx) {
                 const el = zoomPanContainer.querySelector(`.pdf-element[data-element-id='${elementIdx}']`);
                 const svgRect = document.querySelector(`#${widgetId} .svg-layer svg rect[data-element-id='${elementIdx}']`);
                 if (!el) return;

                 el.style.backgroundColor = "rgba(64, 158, 255, 0.15)";
                 el.style.border = "2px solid rgba(64, 158, 255, 0.6)";

                 if (svgRect) {
                    svgRect.setAttribute("stroke", "rgba(64, 158, 255, 0.9)");
                    svgRect.setAttribute("stroke-width", "2.5");
                 }
            }
            
            // --- Background Click Listener (on outer container) ---
            outerContainer.addEventListener('click', (event) => {
                // Ignore click if it resulted from the end of a drag
                if (viewerData.justDragged) {
                    return; 
                }

                // If the click is on an element itself, let the element's click handler manage it.
                if (event.target.closest('.pdf-element')) {
                    return;
                }
                // If dragging, don't deselect
                if (viewerData.isDragging) {
                     return;
                }

                // If an element is selected, deselect it
                if (viewerData.selectedElement !== null) {
                    resetElementStyle(viewerData.selectedElement);
                    viewerData.selectedElement = null;

                    // Optionally clear the info panel
                    const infoPanel = document.getElementById(widgetId + "-info-panel");
                    const elementData = document.getElementById(widgetId + "-element-data");
                    if (infoPanel && elementData) {
                        // infoPanel.style.display = "none"; // Or hide it
                        elementData.textContent = ""; // Clear content
                    }
                }
            });

            // Add click handlers to elements
            elements.forEach(function(el) {
                el.addEventListener("click", function(event) {
                    // Stop propagation to prevent the background click handler from immediately deselecting.
                    event.stopPropagation();
                    
                    const elementIdx = parseInt(this.dataset.elementId);
                    const viewer = window.pdfViewerRegistry[widgetId];

                    // If there was a previously selected element, reset its style
                    if (viewer.selectedElement !== null && viewer.selectedElement !== elementIdx) {
                        resetElementStyle(viewer.selectedElement);
                    }

                    // If clicking the already selected element, deselect it (optional, uncomment if desired)
                    /*
                    if (viewer.selectedElement === elementIdx) {
                         resetElementStyle(elementIdx);
                         viewer.selectedElement = null;
                         // Clear info panel maybe?
                         const elementData = document.getElementById(widgetId + "-element-data");
                         if (elementData) elementData.textContent = '';
                         return; // Stop further processing
                    }
                    */
                    
                    // Store newly selected element
                    viewer.selectedElement = elementIdx;
                    
                    // Highlight newly selected element
                    setElementHighlightStyle(elementIdx);
                    
                    // Update info panel
                    const infoPanel = document.getElementById(widgetId + "-info-panel");
                    const elementData = document.getElementById(widgetId + "-element-data");
                    
                    if (infoPanel && elementData) {
                        const element = viewer.initialData.elements[elementIdx];
                        if (!element) { /* console.error(`[${widgetId}] Element data not found for index ${elementIdx}!`); */ return; }
                        infoPanel.style.display = "block";
                        elementData.textContent = JSON.stringify(element, null, 2);
                    } else {
                        /* console.error(`[${widgetId}] Info panel or element data container not found via getElementById on click!`); */
                    }
                });
                
                // Add hover effects
                el.addEventListener("mouseenter", function() {
                    // *** Only apply hover if NOTHING is selected ***
                    const viewer = window.pdfViewerRegistry[widgetId];
                    if (viewer.selectedElement !== null) {
                        return; // Do nothing if an element is selected
                    }
                    // Avoid hover effect while dragging
                    if (viewer.isDragging) {
                         return;
                    }

                    const elementIdx = parseInt(this.dataset.elementId);

                    // Apply hover styling
                    setElementHighlightStyle(elementIdx);
                    
                    // Show element info on hover (only if nothing selected)
                    const infoPanel = document.getElementById(widgetId + "-info-panel");
                    const elementData = document.getElementById(widgetId + "-element-data");
                    
                    if (infoPanel && elementData) {
                        const element = viewer.initialData.elements[elementIdx];
                        if (!element) { /* console.error(`[${widgetId}] Element data not found for index ${elementIdx}!`); */ return; }
                        infoPanel.style.display = "block";
                        elementData.textContent = JSON.stringify(element, null, 2);
                    } else {
                         // Don't spam console on hover if it's not found initially
                         // console.error(`[${widgetId}] Info panel or element data container not found via getElementById on hover!`); 
                    }
                });
                
                el.addEventListener("mouseleave", function() {
                    // *** Only reset hover if NOTHING is selected ***
                    const viewer = window.pdfViewerRegistry[widgetId];
                    if (viewer.selectedElement !== null) {
                         return; // Do nothing if an element is selected
                    }
                    // Avoid hover effect while dragging
                     if (viewer.isDragging) {
                          return;
                     }

                    const elementIdx = parseInt(this.dataset.elementId);
                    
                    // Reset styling
                    resetElementStyle(elementIdx);

                    // Optionally hide/clear the info panel on mouse leave when nothing is selected
                    // const infoPanel = document.getElementById(widgetId + "-info-panel");
                    // const elementData = document.getElementById(widgetId + "-element-data");
                    // if (infoPanel && elementData) {
                    //     elementData.textContent = '';
                    // }
                });
            });
                        
        })();
        """ % (self.widget_id, json.dumps(self.pdf_data))
        
        # Add the JavaScript
        display(Javascript(js_code))
    
    def _repr_html_(self):
        """Return empty string as HTML has already been displayed"""
        return ""

    @classmethod
    def from_page(cls, page, on_element_click=None, include_attributes=None):
        """
        Create a viewer widget from a Page object.
        
        Args:
            page: A natural_pdf.core.page.Page object
            on_element_click: Optional callback function for element clicks
            include_attributes: Optional list of *additional* specific attributes to include.
                                A default set of common/useful attributes is always included.
            
        Returns:
            SimpleInteractiveViewerWidget instance or None if image rendering fails.
        """
        # Get the page image
        import base64
        from io import BytesIO
        import json # Ensure json is imported
        from PIL import Image # Ensure Image is imported
        
        # Render page to image using the correct method and parameter
        scale = 1.0 # Define scale factor used for rendering
        try:
            img_object = page.to_image(resolution=int(72 * scale)) # Call to_image
            # Check if .original attribute exists, otherwise assume img_object is the PIL Image
            if hasattr(img_object, 'original') and isinstance(img_object.original, Image.Image):
                 img = img_object.original
            elif isinstance(img_object, Image.Image):
                 img = img_object
            else:
                 # If it's neither, maybe it's the raw bytes? Try opening it.
                 try:
                     img = Image.open(BytesIO(img_object)).convert('RGB')
                 except Exception:
                      raise TypeError(f"page.to_image() returned unexpected type: {type(img_object)}")
            logger.debug(f"Successfully rendered page {page.index} using to_image()")
        except Exception as render_err:
            logger.error(f"Error rendering page {page.index} image for widget: {render_err}", exc_info=True)
            # Return None or raise the error? Let's raise for now to make it clear.
            raise ValueError(f"Failed to render page image: {render_err}") from render_err
        
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        image_uri = f"data:image/png;base64,{img_str}"
        
        # Convert elements to dict format
        elements = []
        # Use page.elements directly if available, otherwise fallback to find_all
        page_elements = getattr(page, 'elements', page.find_all('*'))
        
        # Filter out 'char' elements
        filtered_page_elements = [el for el in page_elements if getattr(el, 'type', '').lower() != 'char']
        logger.debug(f"Filtered out char elements, keeping {len(filtered_page_elements)} elements.")
        
        # Define a list of common/useful attributes (properties) to check for
        default_attributes_to_get = [
            'text', 'fontname', 'size', 'bold', 'italic', 'color',
            'linewidth', # For lines (pdfplumber uses 'linewidth')
            'is_horizontal', 'is_vertical', # For lines
            'source', 'confidence', # For text/OCR
            'label', # Common for layout elements
            'model', # Add the model name (engine)
            # Add any other common properties you expect from your elements
            'upright', 'direction' # from pdfplumber chars/words
        ]
        
        for i, element in enumerate(filtered_page_elements):
            # Get original coordinates and calculated width/height (always present via base class)
            original_x0 = element.x0
            original_y0 = element.top
            original_x1 = element.x1
            original_y1 = element.bottom
            width = element.width
            height = element.height
            
            # Base element dict with required info
            elem_dict = {
                'id': i,
                # Use the standardized .type property
                'type': element.type,
                # Scaled coordinates for positioning in HTML/SVG
                'x0': original_x0 * scale,
                'y0': original_y0 * scale,
                'x1': original_x1 * scale,
                'y1': original_y1 * scale,
                'width': width * scale,
                'height': height * scale,
            }
            
            # --- Get Default Attributes --- #
            attributes_found = set()
            for attr_name in default_attributes_to_get:
                if hasattr(element, attr_name):
                    try:
                        value = getattr(element, attr_name)
                        # Convert non-JSON serializable types to string
                        processed_value = value
                        if not isinstance(value, (str, int, float, bool, list, dict, tuple)) and value is not None:
                             processed_value = str(value)
                        elem_dict[attr_name] = processed_value
                        attributes_found.add(attr_name)
                    except Exception as e:
                        logger.warning(f"Could not get or process default attribute '{attr_name}' for element {i} ({element.type}): {e}")
            
            # --- Get User-Requested Attributes (if any) --- #
            if include_attributes:
                 for attr_name in include_attributes:
                     # Only process if not already added and exists
                     if attr_name not in attributes_found and hasattr(element, attr_name):
                         try:
                             value = getattr(element, attr_name)
                             processed_value = value
                             if not isinstance(value, (str, int, float, bool, list, dict, tuple)) and value is not None:
                                  processed_value = str(value)
                             elem_dict[attr_name] = processed_value
                         except Exception as e:
                              logger.warning(f"Could not get or process requested attribute '{attr_name}' for element {i} ({element.type}): {e}")
            for attr_name in elem_dict:
                if isinstance(elem_dict[attr_name], float):
                    elem_dict[attr_name] = round(elem_dict[attr_name], 2)
            elements.append(elem_dict)
        
        logger.debug(f"Prepared {len(elements)} elements for widget with scaled coordinates and curated attributes.")
        
        # Create and return widget
        # The actual JSON conversion happens when the data is sent to the frontend
        return cls(
            image_uri=image_uri,
            elements=elements
        )

# Keep the original widget class for reference, but make it not register
# by commenting out the decorator
# @widgets.register
class InteractiveViewerWidget(widgets.DOMWidget):
    """Jupyter widget for interactively viewing PDF page elements."""
    _view_name = Unicode('InteractiveViewerView').tag(sync=True)
    _view_module = Unicode('viewer_widget').tag(sync=True)
    _view_module_version = Unicode('^0.1.0').tag(sync=True)

    image_uri = Unicode('').tag(sync=True)
    page_dimensions = Dict({}).tag(sync=True)
    elements = List([]).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.debug("InteractiveViewerWidget initialized (Python).")

    # Example observer (optional)
    @observe('elements')
    def _elements_changed(self, change):
        # Only log if logger level allows
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Python: Elements traitlet changed. New count: {len(change['new'])}")
        # Can add Python-side logic here if needed when elements change
        # print(f"Python: Elements traitlet changed. New count: {len(change['new'])}")
        pass

# Example usage
"""
Example usage:

# Method 1: Using pdf_data dictionary
viewer = SimpleInteractiveViewerWidget(pdf_data={
    'page_image': 'data:image/png;base64,...',  # Base64 encoded image
    'elements': [
        {
            'type': 'text', 
            'x0': 100, 
            'y0': 200, 
            'x1': 300, 
            'y1': 220, 
            'text': 'Sample text'
        }
    ]
})

# Method 2: Using keyword arguments
viewer = SimpleInteractiveViewerWidget(
    image_uri='data:image/png;base64,...',  # Base64 encoded image
    elements=[
        {
            'type': 'text', 
            'x0': 100, 
            'y0': 200, 
            'x1': 300, 
            'y1': 220, 
            'text': 'Sample text'
        }
    ]
)

# Method 3: Using a Page object
from natural_pdf.core.page import Page
page = doc.pages[0]  # Assuming 'doc' is a Document object
viewer = SimpleInteractiveViewerWidget.from_page(page)

# Display the widget
viewer
""" 