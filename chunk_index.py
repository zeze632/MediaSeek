"""
长文本切片索引：把 PDF 按段落切片，每段单独编码，存入 ChromaDB
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import chromadb
from PyPDF2 import PdfReader
from FlagEmbedding import BGEM3FlagModel

# 1. 加载模型和 ChromaDB
print('加载 BGE-M3...')
model = BGEM3FlagModel('bge-m3', use_fp16=True)
client = chromadb.PersistentClient(path='./chroma_db')

# 删除旧的切片集合
try:
    client.delete_collection('doc_chunks')
except:
    pass
chunk_collection = client.create_collection(
    name='doc_chunks',
    # 与 multi_modal 保持同一度量，避免两套距离尺度混用
    metadata={'hnsw:space': 'cosine'},
)
print('ChromaDB 切片集合已创建\n')

# 2. 读取 PDF 的标题（从向量库 JSON）
pdf_titles = {}
with open('向量库/db_pdf.json', 'r', encoding='utf-8') as f:
    pdf_data = json.load(f)
for item in pdf_data['items']:
    pdf_titles[item['file_name']] = item['dense_text']

PDF_DIR = '多模态数据/multi_data/pdf/all_pdf'
CHUNK_SIZE = 500  # 每段最多 500 个字

# 3. 处理每个 PDF
total_chunks = 0
for filename, title in pdf_titles.items():
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        print(f'  跳过: {filename}（文件不存在）')
        continue

    # 3.1 提取 PDF 文本
    try:
        reader = PdfReader(pdf_path)
        full_text = ''
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + '\n'
    except Exception as e:
        print(f'  提取失败 {filename}: {e}')
        continue

    if not full_text.strip():
        print(f'  跳过 {filename}（无文本内容）')
        continue

    # 3.2 切片：每 500 字切一段，重叠 50 字避免截断
    chunks = []
    i = 0
    while i < len(full_text):
        chunk = full_text[i:i + CHUNK_SIZE]
        chunks.append(chunk.strip())
        i += CHUNK_SIZE - 50  # 重叠 50 字

    # 3.3 编码入库
    for j, chunk in enumerate(chunks):
        chunk_id = f"{os.path.splitext(filename)[0]}_chunk{j:03d}"
        output = model.encode([chunk], return_dense=True, return_sparse=True)
        chunk_collection.add(
            ids=[chunk_id],
            embeddings=[[float(v) for v in output['dense_vecs'][0]]],
            metadatas=[{
                'file_name': filename,
                'chunk_index': j,
                'total_chunks': len(chunks),
                'chunk_text': chunk[:300],
                'db_type': 'pdf',
            }],
            documents=[chunk],
        )
        total_chunks += 1

    print(f'  {filename}: {len(chunks)} 段（共 {len(full_text)} 字）')

print(f'\nPDF 切片入库完成！共 {total_chunks} 段')
