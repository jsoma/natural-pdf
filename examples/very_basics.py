from natural_pdf import PDF

# Open the PDF
pdf = PDF("./pdfs/01-practice.pdf")

# Approximate match for red
serial = pdf.find('text[color~=red]')

# Between 'Summary' and thick line
summary = pdf.find('text:contains("Summary")').below(include_element=True, until='line[width>=2]')

# Debug
serial.highlight(label='Serial')
summary.highlight(label='Summary')
pdf.pages[0].to_image(path="output.png", show_labels=True)
