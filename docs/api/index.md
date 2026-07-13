# API Reference

This section provides detailed documentation for the public classes and methods in Natural PDF.

## Core Classes

::: natural_pdf
    options:
      show_source: false
      show_bases: true
      inherited_members: true
      members: true

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

::: natural_pdf.text.contracts.TextLayoutOptions
    options:
      show_source: false

::: natural_pdf.text.contracts.ExtractedText
    options:
      show_source: false

::: natural_pdf.text.contracts.SourceTextSegment
    options:
      show_source: false

The less commonly imported extraction hosts are documented explicitly below;
they are not re-exported from the package root, but their signatures are part
of the public contract:

::: natural_pdf.elements.rect.RectangleElement
    options:
      show_source: false
      inherited_members: true

::: natural_pdf.elements.element_collection.ElementCollection
    options:
      show_source: false
      inherited_members: true

::: natural_pdf.flows.collections.FlowElementCollection
    options:
      show_source: false
      inherited_members: true

::: natural_pdf.flows.collections.FlowRegionCollection
    options:
      show_source: false
      inherited_members: true

::: natural_pdf.flows.element.FlowElement
    options:
      show_source: false
      inherited_members: true
