import glob
import pandas as pd
from docx import Document

print("=== README.docx ===")
for p in Document("data/README.docx").paragraphs:
    if p.text.strip():
        print(p.text)

for f in sorted(glob.glob("data/*.xlsx")):
    sheets = pd.read_excel(f, sheet_name=None)
    for name, d in sheets.items():
        print(f"\n=== {f} | sheet '{name}' | shape {d.shape} ===")
        print(d.head(3).to_string())