# API Reference

This section provides detailed documentation for the public classes and methods in Natural PDF.

## Core Classes

<!-- npdf-api:include id=natural-pdf -->

Text extraction methods are inherited from the host-specific text contract
mixins. The [text extraction contract reference](text-extraction.md) documents
the four signatures, supported options, return types, and migration rules.

The API build keeps inherited members enabled so each host's generated page
shows the same canonical signature as the corresponding contract family.

## Public text contract types

The value objects used by text extraction are also available from
`natural_pdf`:

```python
from natural_pdf import (
    ExtractedText,
    SourceTextSegment,
    TextLayoutOptions,
    WhitespaceMode,
)
```

Their complete fields and validation rules are defined in
`natural_pdf.text.contracts`.

<!-- npdf-api:include id=text-layout-options -->

<!-- npdf-api:include id=extracted-text -->

<!-- npdf-api:include id=source-text-segment -->

The less commonly imported extraction hosts are documented explicitly below;
they are not re-exported from the package root, but their signatures are part
of the public contract:

<!-- npdf-api:include id=rectangle-element -->

<!-- npdf-api:include id=element-collection -->

<!-- npdf-api:include id=flow-element-collection -->

<!-- npdf-api:include id=flow-region-collection -->

<!-- npdf-api:include id=flow-element -->
