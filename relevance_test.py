"""
相关性对比实验：比较"传统文化"与"非物质文化遗产"在 Dense vs Sparse 下的检索差异
在终端里逐条打分：输入 2/1/0 然后回车
  2 = 高度相关  1 = 有点关系  0 = 不相关
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import random
import numpy as np
from FlagEmbedding import BGEM3FlagModel

# ===== 1. 加载 =====
print('加载模型和向量库...')
model = BGEM3FlagModel('bge-m3', use_fp16=True)

all_items = []
for db_type in ['image', 'video', 'audio', 'pdf']:
    with open(f'向量库/db_{db_type}.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data['items']:
        item['dense_vector'] = np.array(item['encoding']['dense_vector'])
        item['sparse_vector'] = item['encoding']['sparse_vector']
        item['db_type'] = db_type
    all_items.extend(data['items'])
print(f'加载完成，共 {len(all_items)} 条\n')


# ===== 2. 检索 =====
def search_top3(query):
    q_dense = model.encode([query], return_dense=True, return_sparse=False)
    q_dense_vec = np.array(q_dense['dense_vecs'][0])
    q_sparse = model.encode([query], return_dense=False, return_sparse=True)
    q_sparse_weights = q_sparse['lexical_weights'][0]

    results = []
    for item in all_items:
        d = np.dot(q_dense_vec, item['dense_vector']) / (
            np.linalg.norm(q_dense_vec) * np.linalg.norm(item['dense_vector']))
        s = sum(q_sparse_weights.get(tid, 0) * item['sparse_vector'].get(tid, 0)
                for tid in q_sparse_weights)
        results.append({'item': item, 'dense': d, 'sparse': s})

    dense_top3 = sorted(results, key=lambda x: x['dense'], reverse=True)[:3]
    sparse_top3 = sorted(results, key=lambda x: x['sparse'], reverse=True)[:3]
    return dense_top3, sparse_top3


# ===== 3. 交互打分 =====
def do_test(query):
    d_top3, s_top3 = search_top3(query)

    print('=' * 60)
    print(f'查询: "{query}"')
    print('  输入 2(相关) / 1(有点关系) / 0(不相关) 然后回车')
    print('=' * 60)

    # 把 Dense 和 Sparse 的结果混在一起并打乱顺序——原来是「先 3 条 Dense 再 3 条 Sparse」
    # 的固定顺序，评委一眼就能猜出来源，盲评等于没盲
    all_results = []
    for i, r in enumerate(d_top3, 1):
        all_results.append(('Dense', i, r))
    for i, r in enumerate(s_top3, 1):
        all_results.append(('Sparse', i, r))
    random.shuffle(all_results)

    scores = {}

    for idx, (source, rank, r) in enumerate(all_results, 1):
        item = r['item']
        print(f'\n--- 结果 {idx}/6 ---')
        print(f'内容: {item["dense_text"][:120]}')
        print(f'类型: [{item["db_type"]}] {item["file_name"]}')

        while True:
            try:
                ans = input('你的打分 (2/1/0): ').strip()
                score = int(ans)
                if score in [0, 1, 2]:
                    key = f'{source}{rank}'
                    scores[key] = score
                    break
                else:
                    print('  请输入 2、1 或 0')
            except ValueError:
                print('  请输入 2、1 或 0')

    # 统计
    d_total = sum(scores.get(f'Dense{i}', 0) for i in range(1, 4))
    s_total = sum(scores.get(f'Sparse{i}', 0) for i in range(1, 4))

    print(f'\n{"=" * 60}')
    print(f'打分结果:')
    print(f'  Dense  Top-3: D1={scores.get("Dense1","?")}, D2={scores.get("Dense2","?")}, D3={scores.get("Dense3","?")}  → 总分 {d_total}')
    print(f'  Sparse Top-3: S1={scores.get("Sparse1","?")}, S2={scores.get("Sparse2","?")}, S3={scores.get("Sparse3","?")}  → 总分 {s_total}')
    if d_total > s_total:
        print(f'  → Dense 胜出！语义检索比关键词检索更准')
    elif s_total > d_total:
        print(f'  → Sparse 胜出！关键词检索比语义检索更准')
    else:
        print(f'  → 平手')
    print()


# ===== 4. 开始测试 =====
print('注意：下面的结果已打乱顺序，不标注是 Dense 还是 Sparse 产生的')
print('请根据内容本身来判断相关性，不要猜来源\n')

do_test('传统文化')
do_test('非物质文化遗产')

print('全部完成！把上面两个结果截图，这就是你的实验证据。')
