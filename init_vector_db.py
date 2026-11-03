"""
导入向量到 ChromaDB：从 JSON 文件读取向量，存入 ChromaDB
后续搜索直接查 ChromaDB，不再逐条暴力比对

注意：本步只是「搬运」已经算好的向量，不需要加载 2.2 GB 的 BGE-M3 模型，
      所以克隆仓库后不下载模型也能先把 ChromaDB 建起来。
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import chromadb

# 1. 创建 ChromaDB 客户端（数据存本地 ./chroma_db 文件夹）
print('创建 ChromaDB...')
client = chromadb.PersistentClient(path='./chroma_db')
# 删除旧的集合（如果存在），重新创建
try:
    client.delete_collection('multi_modal')
except:
    pass
collection = client.create_collection(
    name='multi_modal',
    # 用余弦距离：距离 = 1 - 余弦相似度，落在 [0, 2]
    # 不指定的话 HNSW 默认是 L2（欧氏距离的平方），得分换算会出错
    metadata={'hnsw:space': 'cosine'},
)

# 3. 从 JSON 读取数据，逐条写入 ChromaDB
ids = []
embeddings = []
metadatas = []
documents = []

for db_type in ['image', 'video', 'audio', 'pdf']:
    with open(f'向量库/db_{db_type}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data['items']:
        ids.append(item['id'])
        embeddings.append(item['encoding']['dense_vector'])
        metadatas.append({
            'file_name': item['file_name'],
            'db_type': item['type'],
            'dense_text': item['dense_text'][:500],
        })
        documents.append(item['dense_text'])

# 4. 写入 ChromaDB
collection.add(
    ids=ids,
    embeddings=embeddings,
    metadatas=metadatas,
    documents=documents,
)

print(f'ChromaDB 初始化完成！共 {len(ids)} 条数据')
print(f'数据已保存在 ./chroma_db 文件夹')
