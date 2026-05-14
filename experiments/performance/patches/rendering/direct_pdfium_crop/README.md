# Rendering: Direct pypdfium Crop

Hypothesis: cropped renders can avoid full-page rasterization by passing crop
amounts directly to `pypdfium2.PdfPage.render()`, then applying the existing
highlight/legend/resize pipeline.

Risk: medium. Direct crop rendering changes the rasterization boundary, so
output dimensions and antialiasing can differ by a small number of pixels from
the current full-page-render-then-crop path.
