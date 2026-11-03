"""
索骥 MediaSeek · 多模态素材检索与问答系统
BGE-M3 双路召回（ChromaDB 稠密 + BM25 稀疏）→ RRF 融合 → DeepSeek RAG

启动方式: python web_app.py
浏览器打开: http://localhost:5000
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import pandas as pd
import chromadb
from FlagEmbedding import BGEM3FlagModel
from flask import Flask, request, jsonify, send_from_directory, abort
from sklearn.decomposition import PCA

from retrieval import BM25Index, rrf_fuse, tokenize_query

app = Flask(__name__)

# ===== 1. 启动时加载模型和 ChromaDB =====
print('加载 BGE-M3 模型...')
model = BGEM3FlagModel('bge-m3', use_fp16=True)
print('模型加载完成！')

print('连接 ChromaDB...')
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_collection(name='multi_modal')
try:
    chunk_collection = client.get_collection(name='doc_chunks')
    print(f'已加载文档切片集合: {chunk_collection.count()} 段\n')
except:
    chunk_collection = None
    print('(无文档切片集合)\n')

# ===== 2. 加载元数据（用于展示和稀疏检索） =====
FILE_PATHS = {
    'image': '多模态数据/multi_data/images/all_images',
    'video': '多模态数据/multi_data/videos/all_video',
    'audio': '多模态数据/multi_data/audios/all_audios',
    'pdf':   '多模态数据/multi_data/pdf/all_pdf',
}

all_items = []
for db_type in ['image', 'video', 'audio', 'pdf']:
    with open(f'向量库/db_{db_type}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data['items']:
        item['dense_vector'] = np.array(item['encoding']['dense_vector'])
        item['sparse_vector'] = item['encoding']['sparse_vector']
        item['db_type'] = db_type
        item['file_path'] = f"{FILE_PATHS[db_type]}/{item['file_name']}"
    all_items.extend(data['items'])
print(f'元数据加载完成：{len(all_items)} 条\n')

# ===== 3. PCA 降维（用于前端可视化） =====
print('计算 PCA...')
all_vectors = np.stack([item['dense_vector'] for item in all_items])
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(all_vectors)
for i, item in enumerate(all_items):
    item['x'] = float(coords_2d[i][0])
    item['y'] = float(coords_2d[i][1])
print('PCA 降维完成！\n')

# 建立 id → item 的快速索引
items_by_id = {item['id']: item for item in all_items}

# 语义相似度阈值：低于它、且关键词也没命中，就视为「不相关」
MIN_DENSE_SCORE = 0.47
# 关键词阈值：BM25 分低于它算没命中（中文短文本通常 0~15）
MIN_BM25 = 3.0
# RRF 融合常数，60 是论文里的常用值
RRF_K = 60
# 融合权重：(语义, 关键词)。关键词路配比低一些，
# 因为中文短描述上的 BM25 噪音比语义大，实测等权会把 Top5 拉差
RRF_WEIGHTS = (1.0, 0.5)

print('构建 BM25 关键词索引...')
bm25 = BM25Index([item['dense_text'] for item in all_items])
print(f'BM25 索引完成：{len(all_items)} 篇文档\n')


def retrieve(query, top_k=5, filter_type='all', mode='hybrid'):
    """统一检索入口：语义（Chroma）+ 关键词（BM25）→ RRF 融合

    返回 (results, query_vec)，results 每项形如
      {'item': 原始条目, 'dense': 余弦相似度, 'sparse': BM25 分, 'final': 排序分}
    """
    # 查询只编码一次（原来 dense/sparse 分两次调用，等于白算一遍）
    q_out = model.encode([query], return_dense=True, return_sparse=False)
    q_vec = np.array(q_out['dense_vecs'][0])

    candidates = [it for it in all_items
                  if filter_type == 'all' or it['db_type'] == filter_type]
    cand_ids = {it['id'] for it in candidates}

    # --- 语义召回：ChromaDB 取前 50 ---
    where = {'db_type': filter_type} if filter_type != 'all' else None
    n_dense = min(50, max(1, len(candidates)))
    res = collection.query(query_embeddings=[q_vec.tolist()],
                           n_results=n_dense, where=where)
    dense_rank = [cid for cid in res['ids'][0] if cid in cand_ids]
    dense_score = {cid: float(1 - d)
                   for cid, d in zip(res['ids'][0], res['distances'][0])}

    # --- 关键词召回：BM25 对全库打分（不再只在前 20 条里找）---
    bm_all = bm25.scores(tokenize_query(query))
    bm_score = {all_items[i]['id']: bm_all[i] for i in range(len(all_items))}
    order = sorted(range(len(all_items)), key=lambda i: bm_all[i], reverse=True)
    bm_rank = [all_items[i]['id'] for i in order
               if all_items[i]['id'] in cand_ids and bm_all[i] > 0][:50]

    def dense_of(iid):
        """Chroma 前 50 之外的条目补算：向量已归一化，点积就是余弦"""
        if iid in dense_score:
            return dense_score[iid]
        return float(np.dot(q_vec, items_by_id[iid]['dense_vector']))

    # 拒答判断放在查询层面：语义最好的都不够像、关键词也一个没命中，
    # 才认为“这个问题语料里没有”，而不是逐条剔除（逐条剔除会误删相关结果）
    top_dense = dense_of(dense_rank[0]) if dense_rank else 0.0
    # 关键词一路也要在「当前筛选范围」内取最高分。原来取的是全库第一名，
    # filter=image 时会被别的模态的高分救回，导致拒答判断失真。
    top_bm25 = max((bm_all[i] for i, it in enumerate(all_items) if it['id'] in cand_ids),
                   default=0.0)
    if top_dense < MIN_DENSE_SCORE and top_bm25 < MIN_BM25:
        return [], q_vec

    if mode == 'dense':
        ranked = [(i, dense_of(i)) for i in dense_rank]
    elif mode == 'sparse':
        top = max([bm_score.get(i, 0) for i in bm_rank], default=0) or 1.0
        ranked = [(i, bm_score.get(i, 0) / top) for i in bm_rank]
    else:
        fused = rrf_fuse([dense_rank, bm_rank], k=RRF_K, weights=RRF_WEIGHTS)
        max_rrf = sum(RRF_WEIGHTS) / (RRF_K + 1)   # 两路都排第 1 时的融合分
        ranked = [(i, s / max_rrf) for i, s in fused]

    results = []
    for iid, final in ranked[:top_k]:
        results.append({
            'item': items_by_id[iid],
            'dense': dense_of(iid),
            'sparse': float(bm_score.get(iid, 0)),
            'final': float(final),
        })
    return results, q_vec


# ===== 4. API：搜索 =====
@app.route('/search', methods=['POST'])
def search():
    query = request.json.get('query', '')
    mode = request.json.get('mode', 'hybrid')
    filter_type = request.json.get('filter_type', 'all')
    if not query:
        return jsonify({'error': '请输入查询内容'}), 400

    results, q_vec = retrieve(query, top_k=5, filter_type=filter_type, mode=mode)
    q_2d = pca.transform(q_vec.reshape(1, -1))[0]

    return jsonify({
        'results': [{
            'id': r['item']['id'],
            'file_name': r['item']['file_name'],
            'db_type': r['item']['db_type'],
            'file_path': r['item']['file_path'],
            'dense_text': r['item']['dense_text'][:150],
            'dense_score': round(r['dense'], 4),
            'sparse_score': round(r['sparse'], 4),
            'final_score': round(r['final'], 4),
        } for r in results],
        'query_2d': {'x': float(q_2d[0]), 'y': float(q_2d[1])},
        'mode': mode,
    })


# ===== 5. API：向量坐标（给前端画散点图） =====
@app.route('/vectors', methods=['GET'])
def get_vectors():
    items_data = []
    for item in all_items:
        items_data.append({
            'id': item['id'],
            'file_name': item['file_name'],
            'db_type': item['db_type'],
            'x': item['x'],
            'y': item['y'],
        })
    return jsonify({'items': items_data})


# ===== 6. 首页 =====
@app.route('/')
def index():
    return send_from_directory('.', 'search_page.html')


# ===== 7. RAG（也使用 ChromaDB） =====
from openai import OpenAI


def _load_api_key():
    """优先读环境变量 DEEPSEEK_API_KEY，其次读项目根目录的 api_key.txt"""
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_key.txt')
    if os.path.exists(key_file):
        with open(key_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return ''


DEEPSEEK_API_KEY = _load_api_key()
if not DEEPSEEK_API_KEY:
    print('[警告] 未找到 DeepSeek API Key：请设置环境变量 DEEPSEEK_API_KEY，'
          '或在项目根目录创建 api_key.txt 写入 Key')

@app.route('/rag', methods=['POST'])
def rag_answer():
    query = request.json.get('query', '')
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    results, q_vec = retrieve(query, top_k=6)
    # 主检索结果：分数是 RRF 归一化分（0~1）
    main_cands = [{'fname': r['item']['file_name'], 'dtype': r['item']['db_type'],
                   'fpath': r['item']['file_path'],
                   'text': r['item']['dense_text'][:500],
                   'score': round(float(r['final']), 4)}
                  for r in results]

    # 长文档切片单独一路：它的分数是裸余弦（通常 0.5~0.8），和 RRF 归一化分不是一个尺度，
    # 直接混排会被主检索的高分全部挤掉（原来的写法就是这样，切片等于白建），所以两条路各自留名额。
    chunk_cands = []
    if chunk_collection:
        cr = chunk_collection.query(query_embeddings=[q_vec.tolist()], n_results=8)
        for i, cid in enumerate(cr['ids'][0]):
            meta = cr['metadatas'][0][i]; doc = cr['documents'][0][i]
            d = float(1 - cr['distances'][0][i])
            if d < MIN_DENSE_SCORE:
                continue
            chunk_cands.append({
                'fname': meta.get('file_name', ''), 'dtype': 'pdf',
                'fpath': f"多模态数据/multi_data/pdf/all_pdf/{meta.get('file_name', '')}",
                'text': doc[:800], 'score': round(d, 4),
            })

    if chunk_cands:
        picks = main_cands[:4] + chunk_cands[:2]      # 4 条主检索 + 2 段 PDF 正文
    else:
        picks = main_cands[:6]

    # 没有任何相关结果就直说，不调用大模型硬编
    if not picks:
        return jsonify({
            'query': query,
            'answer': '资料里没有找到与该问题相关的内容，换个说法或换个关键词再试试。',
            'references': [],
        })

    context_lines = [f"[{i+1}] ({r['dtype']}) {r['fname']}\n   内容: {r['text'][:300]}" for i, r in enumerate(picks)]
    context = "\n".join(context_lines)

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=[
                {"role": "system", "content": "你是一个检索助手。根据用户提供的参考资料回答用户的问题。"
                 "如果用户在提问中带有格式要求（如'只要视频'、'只要图片'、'3条'等），请按用户要求筛选和整理结果。"
                 "如果资料不足以回答，就说不知道，不要编造。"},
                {"role": "user", "content": f"参考资料：\n{context}\n\n用户问题：{query}"}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"调用 AI 失败: {str(e)}"

    return jsonify({
        'query': query, 'answer': answer,
        'references': [{'file_name': r['fname'], 'db_type': r['dtype'], 'file_path': r['fpath'],
                        'score': round(float(r['score']), 4)} for r in picks],
    })


# ===== 8. AI 多轮对话（带记忆） =====
chat_sessions = {}  # {session_id: [{"role": "user"/"assistant", "content": "..."}]}
MAX_CHAT_SESSIONS = 200   # 最多同时保留的会话数
MAX_CHAT_TURNS = 8        # 每个会话保留的轮数（一问一答算一轮）

@app.route('/chat', methods=['POST'])
def chat():
    query = request.json.get('query', '')
    session_id = request.json.get('session_id', 'default')
    if not query:
        return jsonify({'error': '请输入问题'}), 400

    # 1. 检索（与检索页同一套混合检索）
    results, _ = retrieve(query, top_k=3)
    context_lines = []
    refs = []
    for r in results:
        item = r['item']
        context_lines.append(f"[资料] {item['dense_text'][:300]}")
        refs.append({'file_name': item['file_name'], 'db_type': item['db_type'],
                     'file_path': item['file_path'], 'score': round(r['final'], 4)})

    # 2. 获取/创建对话历史（会话数有上限，避免长期运行内存只涨不降）
    if session_id not in chat_sessions:
        if len(chat_sessions) >= MAX_CHAT_SESSIONS:
            chat_sessions.pop(next(iter(chat_sessions)))   # 淘汰最早建立的会话
        chat_sessions[session_id] = []
    history = chat_sessions[session_id]
    if len(history) > MAX_CHAT_TURNS * 2:
        history = history[-MAX_CHAT_TURNS * 2:]

    # 检索不到相关内容就直接回答，不调大模型
    if not context_lines:
        answer = '资料里没有找到与该问题相关的内容，换个说法或换个关键词再试试。'
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        chat_sessions[session_id] = history
        return jsonify({'answer': answer, 'references': [], 'session_id': session_id})

    # 3. 构建消息
    messages = [
        {"role": "system", "content": "你是检索助手，根据资料和对话历史回答。如果资料不足就说不知道。"},
    ]
    # 添加最近6轮对话（保持上下文）
    messages.extend(history[-6:])
    # 添加当前问题（带资料）
    context = "\n".join(context_lines[:3])
    messages.append({"role": "user", "content": f"参考资料：\n{context}\n\n问题：{query}"})

    # 4. 调 DeepSeek
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-v4-flash", messages=messages, temperature=0.3, max_tokens=500)
        answer = resp.choices[0].message.content
    except Exception as e:
        answer = f"调用失败: {str(e)}"

    # 5. 保存对话记录
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    chat_sessions[session_id] = history

    return jsonify({'answer': answer, 'references': refs[:3], 'session_id': session_id})

@app.route('/chat')
def chat_page():
    return send_from_directory('.', 'chat.html')

@app.route('/admin')
def admin_page():
    return send_from_directory('.', 'admin.html')


PDF_CHUNK_SIZE = 500
PDF_CHUNK_OVERLAP = 50


def index_pdf_chunks(pdf_path, file_name):
    """把单个 PDF 切片写进 ChromaDB，让 RAG 能检索到正文。

    原来上传 PDF 只更新了主库（标题），正文永远搜不到，得手动跑 chunk_index.py。
    """
    global chunk_collection
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(pdf_path)
        full_text = ''.join((page.extract_text() or '') + '\n' for page in reader.pages)
    except Exception as e:
        print(f'  PDF 文本提取失败: {e}')
        return 0
    if not full_text.strip():
        print('  PDF 没有文本层，跳过切片')
        return 0

    if chunk_collection is None:
        try:
            chunk_collection = client.create_collection(
                'doc_chunks', metadata={'hnsw:space': 'cosine'})
        except Exception:
            chunk_collection = client.get_collection('doc_chunks')

    stem = os.path.splitext(file_name)[0]
    chunks, i = [], 0
    while i < len(full_text):
        piece = full_text[i:i + PDF_CHUNK_SIZE].strip()
        if piece:
            chunks.append(piece)
        i += PDF_CHUNK_SIZE - PDF_CHUNK_OVERLAP

    for j, chunk in enumerate(chunks):
        out = model.encode([chunk], return_dense=True, return_sparse=False)
        chunk_collection.upsert(   # 同名 PDF 重传时覆盖，不报重复 ID
            ids=[f'{stem}_chunk{j:03d}'],
            embeddings=[[float(v) for v in out['dense_vecs'][0]]],
            metadatas=[{'file_name': file_name, 'chunk_index': j,
                        'total_chunks': len(chunks), 'chunk_text': chunk[:300],
                        'db_type': 'pdf'}],
            documents=[chunk],
        )
    print(f'  PDF 切片入库: {file_name} → {len(chunks)} 段')
    return len(chunks)


@app.route('/upload', methods=['POST'])
def upload_data():
    """上传文件 + 描述 → 编码 → 入库"""
    file = request.files.get('file')
    db_type = request.form.get('type', 'image')
    main_desc = request.form.get('main_desc', '').strip()
    sub_desc = request.form.get('sub_desc', '').strip()

    if not file or not main_desc:
        return jsonify({'error': '请上传文件并填写主描述'}), 400

    # 类型校验：避免 FILE_PATHS[db_type] 直接 KeyError，也防止传错模态
    if db_type not in FILE_PATHS:
        return jsonify({'error': f'不支持的类型: {db_type}'}), 400

    MAX_UPLOAD = 200 * 1024 * 1024  # 200 MB
    if request.content_length and request.content_length > MAX_UPLOAD:
        return jsonify({'error': '文件不能超过 200 MB'}), 413

    ext = os.path.splitext(file.filename)[1].lower()
    ALLOWED_EXT = {
        'image': {'.jpg', '.jpeg', '.png', '.webp'},
        'video': {'.mp4', '.mov', '.avi', '.mkv'},
        'audio': {'.mp3', '.wav', '.m4a', '.flac'},
        'pdf':   {'.pdf'},
    }
    if ext not in ALLOWED_EXT[db_type]:
        return jsonify({
            'error': f'{db_type} 只接受 {", ".join(sorted(ALLOWED_EXT[db_type]))}，收到 {ext or "(无扩展名)"}'
        }), 400

    # 1. 保存文件
    text = f"{main_desc} {sub_desc}".strip() if sub_desc else main_desc
    save_dir = FILE_PATHS[db_type]
    os.makedirs(save_dir, exist_ok=True)

    # 命名：取现有同前缀文件的最大编号 +1，避免用「文件个数+1」导致重名覆盖
    name_prefix = {'image': 'image', 'video': 'video', 'audio': 'audio', 'pdf': 'doc'}[db_type]
    max_num = 0
    for fname in os.listdir(save_dir):
        stem = os.path.splitext(fname)[0]
        if stem.startswith(name_prefix) and stem[len(name_prefix):].isdigit():
            max_num = max(max_num, int(stem[len(name_prefix):]))
    next_num = max_num + 1
    save_name = f'{name_prefix}{next_num}{ext}'

    save_path = os.path.join(save_dir, save_name)
    while os.path.exists(save_path):   # 并发或残留文件时顺延编号，避免覆盖已有媒体
        next_num += 1
        save_name = f'{name_prefix}{next_num}{ext}'
        save_path = os.path.join(save_dir, save_name)
    file.save(save_path)

    # 2. BGE-M3 编码
    output = model.encode([text], return_dense=True, return_sparse=True)
    dense_vec = [float(v) for v in output['dense_vecs'][0]]
    sparse_vec = {int(k): float(v) for k, v in output['lexical_weights'][0].items()}

    # 3. 生成 ID
    type_map = {'image': 'img', 'video': 'video', 'audio': 'audio', 'pdf': 'pdf'}
    prefix = type_map[db_type]

    # 从现有 JSON 获取最大序号
    existing_ids = [item['id'] for item in all_items if item['id'].startswith(prefix)]
    max_num = 0
    for eid in existing_ids:
        try:
            num = int(eid.split('_')[1])
            if num > max_num:
                max_num = num
        except:
            pass
    new_id = f"{prefix}_{max_num + 1:03d}"

    # 4. 更新 JSON 向量库
    json_file = f'向量库/db_{db_type}.json'
    with open(json_file, 'r', encoding='utf-8') as f:
        db_data = json.load(f)

    new_item = {
        "id": new_id,
        "file_name": save_name,
        "type": db_type,
        "dense_text": text,
        "sparse_text": text,
        "encoding": {
            "model": "bge-m3",
            "dense_vector": dense_vec,
            "sparse_vector": sparse_vec
        }
    }
    db_data['items'].append(new_item)
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(db_data, f, ensure_ascii=False, indent=2)

    # 5. 加入 ChromaDB
    collection.add(
        ids=[new_id],
        embeddings=[dense_vec],
        metadatas=[{'file_name': save_name, 'db_type': db_type, 'dense_text': text[:500]}],
        documents=[text],
    )

    # 5.1 PDF 额外切片入库，否则 RAG 只能看到标题、看不到正文
    if db_type == 'pdf':
        index_pdf_chunks(save_path, save_name)

    # 6. 更新内存中的数据
    new_item_full = {
        'id': new_id,
        'file_name': save_name,
        'db_type': db_type,
        'file_path': f"{FILE_PATHS[db_type]}/{save_name}",
        'dense_text': text,
        'sparse_text': text,
        'dense_vector': np.array(dense_vec),
        'sparse_vector': sparse_vec,
    }
    all_items.append(new_item_full)
    items_by_id[new_id] = new_item_full

    # 更新 PCA（简单追加，不重新计算全部）
    new_2d = pca.transform(np.array(dense_vec).reshape(1, -1))[0]
    new_item_full['x'] = float(new_2d[0])
    new_item_full['y'] = float(new_2d[1])

    # 7. 更新 Excel
    excel_map = {
        'image': '多模态数据/multi_data/images/Image_annotation.xlsx',
        'video': '多模态数据/multi_data/videos/video_annotation.xlsx',
        'audio': '多模态数据/multi_data/audios/audio-annotation.xlsx',
        'pdf': '多模态数据/multi_data/pdf/pdf_annotation.xlsx',
    }
    excel_path = excel_map[db_type]
    if db_type == 'image':
        # 图片表没有表头（第 0 行就是 image2.jpg），不能 skiprows，否则会永久删掉第一行
        df = pd.read_excel(excel_path, header=None)
        new_row = pd.DataFrame([[save_name, '', '', '', '', db_type, main_desc, '', '', sub_desc]])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_path, index=False, header=False)
    elif db_type in ('video', 'audio'):
        df = pd.read_excel(excel_path)
        row = {'file_name': [save_name], 'type': [db_type],
               'main_desc': [main_desc], 'sub_desc': [sub_desc]}
        if db_type == 'video':
            # 视频表比音频表多一列 main_character，漏写这列会让重建索引时文本变成 "nan: 描述"
            row['main_character'] = ['']
        df = pd.concat([df, pd.DataFrame(row)], ignore_index=True)
        df.to_excel(excel_path, index=False)
    else:  # pdf
        df = pd.read_excel(excel_path)
        new_row = pd.DataFrame({'file_name': [save_name], 'title': [main_desc]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(excel_path, index=False)

    print(f'  新数据入库: {new_id} ({save_name})')
    return jsonify({'success': True, 'id': new_id, 'file_name': save_name})


ALLOWED_FILE_ROOT = '多模态数据'


@app.route('/files/<path:filepath>')
def serve_file(filepath):
    """只允许读取多模态数据目录，避免源码、密钥、向量库被下载"""
    target = os.path.realpath(os.path.join('.', filepath))
    root = os.path.realpath(ALLOWED_FILE_ROOT)
    if not target.startswith(root + os.sep):
        abort(403)
    return send_from_directory('.', filepath)


# ===== 9. 启动 =====
if __name__ == '__main__':
    print('=' * 60)
    print('索骥 MediaSeek · 多模态素材检索与问答系统')
    print('浏览器打开: http://localhost:5000')
    print(f'向量数据库: ChromaDB ({collection.count()} 条, 已建索引)')
    print('=' * 60)
    app.run(host='127.0.0.1', port=5000, debug=False)
