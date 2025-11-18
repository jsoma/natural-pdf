## Extending Natural PDF

Natural PDF exposes every capability (selectors, OCR, QA, tables, navigation, etc.) through small helper functions that call into services. Extending the library means swapping one of those services or adding a new helper + service pair. This guide focuses on the “happy path”: how to hook into the runtime with the least ceremony.

---

### Quickstart: override selector behavior

```python
from natural_pdf.core.context import PDFContext
from natural_pdf.services.selector_service import SelectorService

class RegexSelector(SelectorService):
    def find(self, host, **kwargs):
        kwargs.setdefault("engine", "regex")
        return super().find(host, **kwargs)


context = PDFContext(selector_service=RegexSelector)
pdf = open_document("invoice.pdf", context=context)
pdf.pages[0].find_all(".item-row")
```

Two lines of code (subclass + context) change how every host performs selector queries.

---

### Quickstart: add a brand-new capability (four lines)

```python
from natural_pdf.services.delegates import register_capability

def summarize(self, *, max_tokens=500):
    return (self.extract_text() or "")[:max_tokens]

register_capability("summarize", summarize)

pdf = PDF("invoice.pdf")
pdf.pages[0].summarize(max_tokens=120)
```

`register_capability` binds the helper to every ServiceHost (Page, Region, PDF, collections, etc.), so once you declare the function, every object picks it up instantly.

---

### Core building blocks

- **Service hosts** – Everything inheriting `ServiceHostMixin` (Page, Region, Flow, collections) stores a `_services` map. The map is filled by `attach_capability` during initialization or by the PDF context when a document is opened.
- **Services** – Modules under `natural_pdf/services/` (selector, table, ocr, qa, navigation, etc.). Each method receives the host plus any user arguments.
- **Helper modules** – `natural_pdf/services/methods/<capability>_methods.py` define the public API. Helpers do the docstring/signature work and call `resolve_service(self, "<capability>")`.
- **Protocols** – Contracts in `natural_pdf/core/interfaces.py` (`SupportsSections`, `SupportsGeometry`, etc.). Services only depend on these protocols, so you can offer new host types by satisfying the protocol instead of subclassing a specific class.
- **PDFContext** – The dependency injection container. Pass it to `open_document(..., context=...)` to provide alternate services, config values, engine factories, or caches.

---

### Adding or customizing behavior

1. **Reuse a helper**
   Need a host to expose an existing capability? Import the helper and assign it:
   ```python
   from natural_pdf.services.methods import ocr_methods
   MyHost.apply_ocr = ocr_methods.apply_ocr
   ```
   As long as `MyHost` satisfies the protocols OCRService expects (e.g., has a `page` and `bbox`), the helper will work.

2. **Wrap a helper for custom return types**
   Flow objects return `FlowElementCollection` instead of `ElementCollection`. Use `@delegate_signature` to keep the canonical signature while altering the return value:
   ```python
   from natural_pdf.selectors.host_mixin import SelectorHostMixin, delegate_signature
   from natural_pdf.services.methods import flow_selector_methods

   class Flow(...):
       @delegate_signature(flow_selector_methods.find)
       def find(self, *args, **kwargs):
           result = flow_selector_methods.find(self, *args, **kwargs)
           return FlowElement.from_physical(result)
   ```

3. **Attach custom hooks**
   Many services look for optional host attributes (e.g., `_ocr_element_manager`, `_qa_metadata`). Override these in your host to steer behavior without editing the service.

4. **Create a new service/helper pair**
   - Define a helper next to existing ones (`services/methods/<capability>_methods.py`).
   - Implement the service in `services/<capability>_service.py`.
   - Register the service in the context (`PDFContext(<capability>_service=MyService)`).
   - Assign the helper to whichever hosts should expose the public method.

---

### Working with collections and flows

- Use the collection helper modules (e.g., `qa_methods.ask_collection`, `table_methods.extract_tables`) to expose batch operations. Collections generally forward to the same service as their individual elements.
- Flow objects should convert physical elements using the provided adapters (`FlowElement.from_physical`, `FlowElementCollection.from_physical`) rather than re-implementing selectors or navigation.
- Guides and analyzers rely on services as well; e.g., `Guides._extract_with_table_service` simply calls `table_methods.extract_table`. If you add a new host type, make sure it exposes the same helper so Guides automatically support it.

---

### Where to look in the codebase

- `natural_pdf/services/base.py` – `resolve_service`, `attach_capability`, and the `ServiceRegistry`.
- `natural_pdf/services/methods/` – canonical helpers for selector, OCR, QA, tables, navigation, classification, etc.
- `natural_pdf/core/page.py`, `natural_pdf/elements/region.py`, `natural_pdf/flows/flow.py` – concrete examples of hosts wiring helpers.
- `natural_pdf/core/interfaces.py` – the protocols that services depend on.

If you encounter a capability that still uses legacy mixins or doesn’t route through a helper, feel free to modernize it: move the logic to a service, expose a helper, and register the method on each host. Once you do that, every user automatically benefits from the centralized implementation and you only have one place to edit when requirements change.
