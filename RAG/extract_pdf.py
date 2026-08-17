from pypdf import PdfReader
import os

pdf_files = [
    "course_materials/1-RAG架构与原理详解.pdf",
    "course_materials/2-RAG实践（基于LangChain）.pdf"
]

for pdf_path in pdf_files:
    reader = PdfReader(pdf_path)
    output_path = pdf_path.replace(".pdf", "_extracted.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"===== {os.path.basename(pdf_path)} =====\n")
        f.write(f"总页数: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            f.write(f"----- 第 {i+1} 页 -----\n")
            f.write(text + "\n\n")
    print(f"已提取: {output_path} ({len(reader.pages)}页)")
