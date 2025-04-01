// natural_pdf/widgets/frontend/viewer.js
// Minimal version for debugging module loading

(function() {
    // Use a flag to prevent multiple definitions if this script runs multiple times
    if (window.interactiveViewerWidgetDefined) {
        console.log("[DEBUG] viewer_widget already defined. Skipping re-definition.");
        // If it was already defined, maybe trigger a manual load if require is available?
        // This is tricky because the initial load might have failed partially.
        if (typeof require !== 'undefined') {
             console.log("[DEBUG] Attempting require(['viewer_widget'])...");
             try {
                 require(['viewer_widget'], function(module) {
                     console.log("[DEBUG] Manual require succeeded:", module);
                 }, function(err) {
                     console.error("[DEBUG] Manual require failed:", err);
                 });
             } catch (e) {
                 console.error("[DEBUG] Error during manual require:", e);
             }
        }
        return;
    }
    window.interactiveViewerWidgetDefined = true;
    console.log("[DEBUG] Defining viewer_widget module for the first time...");

    // Check for requirejs *after* setting the flag, before defining
     if (typeof requirejs === 'undefined') {
         console.error('[DEBUG] requirejs is still not defined. Widget frontend cannot load.');
         // Maybe display an error in the widget area itself?
         // This suggests a fundamental issue with the Jupyter environment setup.
         return;
     }
     if (typeof define !== 'function' || !define.amd) {
          console.error('[DEBUG] define is not a function or define.amd is missing. Cannot define module.');
          return;
     }

    // Clear any previous potentially failed definition
    require.undef('viewer_widget');

    // Define the module
    define('viewer_widget', ['@jupyter-widgets/base'], function(widgets) {
        console.log("[DEBUG] viewer_widget define callback executed.");
        console.log("[DEBUG] @jupyter-widgets/base loaded:", widgets);

        // Define a very simple view class
        class InteractiveViewerView extends widgets.DOMWidgetView {
            render() {
                console.log("[DEBUG] InteractiveViewerView: render() called.");
                this.el.textContent = 'Minimal Widget Loaded!'; // Simple text content
                this.el.style.border = '2px solid green';
                this.el.style.padding = '10px';

                // Log received data
                this.model.on('change:image_uri', () => console.log("[DEBUG] image_uri changed:", this.model.get('image_uri') ? 'Present' : 'Empty'), this);
                this.model.on('change:page_dimensions', () => console.log("[DEBUG] page_dimensions changed:", this.model.get('page_dimensions')), this);
                this.model.on('change:elements', () => console.log("[DEBUG] elements changed:", this.model.get('elements').length), this);

                 // Log initial data
                 console.log("[DEBUG] Initial image_uri:", this.model.get('image_uri') ? 'Present' : 'Empty');
                 console.log("[DEBUG] Initial page_dimensions:", this.model.get('page_dimensions'));
                 console.log("[DEBUG] Initial elements count:", this.model.get('elements').length);
            }

            remove() {
                console.log("[DEBUG] InteractiveViewerView: remove() called.");
                super.remove();
            }
        }

        console.log("[DEBUG] viewer_widget module definition returning view.");
        // Return the view class
        return {
            InteractiveViewerView: InteractiveViewerView
        };
    }, function(err) {
         // Error callback for the define function
         console.error("[DEBUG] Error loading module dependencies:", err);
         const failedId = err.requireModules && err.requireModules[0];
         if (failedId === 'react' || failedId === 'react-dom' || failedId === 'htm') {
             console.error(`[DEBUG] Failed to load CDN dependency: ${failedId}. Check network connection and CDN availability.`);
         } else if (failedId === '@jupyter-widgets/base') {
              console.error("[DEBUG] Failed to load @jupyter-widgets/base. Ensure ipywidgets frontend is installed and enabled.");
         }
    });

})(); 