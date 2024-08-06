from pdfquery import PDFQuery

pdf = PDFQuery('Accident_investigation.pdf')
pdf.load()

# Use CSS-like selectors to locate the elements
text_elements = pdf.pq('LTTextLineHorizontal')

# Extract the text from the elements
text = [t.text for t in text_elements]
combined_text = " ".join(text)

print(combined_text)