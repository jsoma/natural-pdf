## Extending Natural PDF

Natural PDF is designed to be extensible. You can plug in custom engines for OCR, table extraction, and more, or replace entire services with your own implementations when you need deeper control.

---

### Registering Custom Engines

Natural PDF uses an `EngineProvider` to manage pluggable backends for OCR, table extraction, and other heavy tasks. You can register your own engines to be used by the standard service methods.

#### Example: Custom OCR Engine

```python
from natural_pdf.engine_provider import get_provider

def my_ocr_factory(context, **kwargs):
    # Return your custom OCR engine instance
    return MyCustomOCREngine(**kwargs)

# Register the engine
get_provider().register("ocr", "my_custom_engine", my_ocr_factory)

# Use it
pdf.pages[0].apply_ocr(engine="my_custom_engine")
```

#### Example: Custom Table Extraction Strategy

```python
from natural_pdf.engine_provider import get_provider

def my_table_engine_factory(context, **kwargs):
    return MyTableEngine(**kwargs)

get_provider().register("table", "custom_strategy", my_table_engine_factory)

# Use it
pdf.pages[0].extract_tables(method="custom_strategy")
```

---

### Replacing Core Services

For deep customization, you can replace entire services by subclassing them and injecting them via `PDFContext`.

```python
from natural_pdf.core.context import PDFContext
from natural_pdf.services.table_service import TableService

class AdvancedTableService(TableService):
    def extract_table(self, host, **kwargs):
        # Custom logic before/after
        print("Extracting table...")
        return super().extract_table(host, **kwargs)

# Inject the custom service
context = PDFContext(table_service=AdvancedTableService)
pdf = PDF("document.pdf", context=context)
```

---

### Core Architecture

- **Service Hosts**: Classes like `Page` and `PDF` hold a `self.services` namespace.
- **Services**: Logic lives in `natural_pdf/services/`.
- **EngineProvider**: Manages low-level engines (Tesseract, TATR, etc.).

This separation allows you to swap out implementations at multiple levels:
1.  **Method Level**: Monkey-patch new helpers.
2.  **Engine Level**: Register new backends for existing services.
3.  **Service Level**: Replace the entire service orchestration.
