import pdfplumber
pdf = pdfplumber.open(r"Seating Plan Sessional II Spring 2026.pdf")
print("pages", len(pdf.pages))
for i in [0,1,2]:
    if i >= len(pdf.pages):
        break
    p = pdf.pages[i]
    print("\n=== page", i+1, "===")
    try:
        txt = p.extract_text() or ""
        print("text_len", len(txt))
        print(txt[:700])
    except Exception as e:
        print("text_err", e)
    try:
        tables = p.extract_tables() or []
        print("tables", len(tables))
        if tables:
            t = tables[0]
            print("first_table_rows", len(t))
            for r in t[:6]:
                print(r)
    except Exception as e:
        print("table_err", e)
pdf.close()
