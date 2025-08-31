# Missing First/Last Columns in guides.extract_table()

## Problem Description

When using `guides.extract_table()`, the first and last columns may be missing from the extracted table. This happens because the `Guides.from_lines()` method by default does not include the page boundaries (x=0 and x=page.width) as vertical guides.

### Example of the Issue

```python
# Default behavior - may miss boundary columns
guides = Guides.from_lines(page)
result = guides.extract_table()
# First column "OFFICER" and last column may be missing
```

## Root Cause

The `from_lines()` method detects lines in the PDF but doesn't automatically add guides at the page boundaries. If your table's first column starts at x=0 or the last column ends at x=page.width, and there are no explicit vertical lines at these positions, those columns won't have guides and will be excluded from extraction.

## Solutions

### Solution 1: Use the `outer` parameter (Recommended)

The simplest fix is to use the `outer=True` parameter when creating guides:

```python
# Include outer boundaries when detecting lines
guides = Guides.from_lines(page, outer=True)
result = guides.extract_table()
```

### Solution 2: Use `include_outer_boundaries` in extract_table

If you've already created guides, you can include boundaries during extraction:

```python
# Create guides normally
guides = Guides.from_lines(page)

# Include boundaries during extraction
result = guides.extract_table(include_outer_boundaries=True)
```

### Solution 3: Manually add boundary guides

For more control, you can manually add guides at the page boundaries:

```python
# Create guides
guides = Guides.from_lines(page)

# Add page boundaries
guides.vertical.add([0, page.width])

# Extract table
result = guides.extract_table()
```

### Solution 4: Create guides from specific positions

If you know the exact column positions:

```python
# Create guides with specific positions including boundaries
guides = Guides(page)
guides.vertical.add([0, 100, 200, 300, 400, page.width])
guides.horizontal.from_lines(page)  # Get horizontal guides from lines

result = guides.extract_table()
```

## Best Practices

1. **Always use `outer=True`** when you expect table content at page boundaries:
   ```python
   guides = Guides.from_lines(page, outer=True)
   ```

2. **Check your guides** before extraction:
   ```python
   guides = Guides.from_lines(page)
   print(f"Vertical guides: {guides.vertical.data}")
   print(f"Page width: {page.width}")

   # Check if boundaries are included
   has_left = 0 in guides.vertical.data
   has_right = page.width in guides.vertical.data
   ```

3. **Visualize guides** to debug issues:
   ```python
   # Show the page with guides overlaid
   guides.show()
   ```

## Complete Example

```python
from natural_pdf import PDF
from natural_pdf.analyzers import Guides

# Load PDF
pdf = PDF("document.pdf")
page = pdf[0]

# Method 1: Best practice - use outer=True
guides = Guides.from_lines(page, outer=True)
table = guides.extract_table()
df = table.to_df()
print(df)

# Method 2: Alternative - use include_outer_boundaries
guides = Guides.from_lines(page)
table = guides.extract_table(include_outer_boundaries=True)
df = table.to_df()
print(df)

# Method 3: Manual control
guides = Guides.from_lines(page)
if 0 not in guides.vertical.data:
    guides.vertical.add([0])
if page.width not in guides.vertical.data:
    guides.vertical.add([page.width])
table = guides.extract_table()
df = table.to_df()
print(df)
```

## When This Issue Occurs

This issue typically occurs when:
- Tables are designed with no margins (content starts at x=0)
- Tables span the full page width
- PDF generators don't include explicit border lines at page edges
- Content is positioned exactly at page boundaries

## Verification

To verify if this is your issue:

```python
# Check text positions
texts = page.find_all('text')
min_x = min(t.x0 for t in texts)
max_x = max(t.x1 for t in texts)

print(f"Text spans from x={min_x} to x={max_x}")
print(f"Page width: {page.width}")

# Check guides
guides = Guides.from_lines(page)
print(f"First guide: {guides.vertical.data[0] if guides.vertical.data else 'None'}")
print(f"Last guide: {guides.vertical.data[-1] if guides.vertical.data else 'None'}")

# If min_x < first guide or max_x > last guide, you need boundaries
```
