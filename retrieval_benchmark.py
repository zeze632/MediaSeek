"""
检索性能基准测试：对比 Dense / Sparse / Hybrid 三种方式

和 web_app.py 共用同一个 retrieve()，所以这里的数字和线上页面一致：
- 混合检索走 RRF 排名融合，不是 0.5*dense + 0.5*sparse 的裸加权
  （余弦和 BM25 点积量纲不同，裸加权相加没有可比性）
- 候选池是 Chroma 前 50 + BM25 全库打分，不是只在前 20 条里挑
- Top-3 重叠度按 id 比较，不再按分数比较（分数几乎不会相同，按分数算恒为 0%）

跑法：python retrieval_benchmark.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from web_app import retrieve

TOP_K = 3
MODES = [('dense', 'Dense'), ('sparse', 'Sparse'), ('hybrid', 'Hybrid')]

# 测试查询集（覆盖不同场景）
TEST_CASES = [
    ('非遗', ['非遗工艺', '传统技法', '广彩瓷']),
    ('广东政策', ['省政府', '行政', '通知']),
    ('交通执法', ['交警', '交通违法', '执法不规范']),
    ('习近平', ['讲话', '纪念会', '两岸关系']),
    ('粤港澳大湾区', ['大湾区', '协同发展', '规划']),
    ('传统文化', ['传统技艺', '非遗', '文化展示']),
]


def main():
    rows = {}
    print('=' * 88)
    print(f'{"查询":<12}{"方式":<8}{"耗时(ms)":>10}{"命中":>6}  Top-1')
    print('-' * 88)

    for query, keywords in TEST_CASES:
        rows[query] = {}
        for mode, label in MODES:
            t0 = time.perf_counter()
            results, _ = retrieve(query, top_k=TOP_K, mode=mode)
            dt = (time.perf_counter() - t0) * 1000

            ids = [r['item']['id'] for r in results]
            top1 = results[0]['item'] if results else None
            hit = bool(top1) and any(kw in top1['dense_text'] for kw in keywords)
            rows[query][label] = {
                'ms': round(dt, 1),
                'ids': ids,
                'top1': top1['file_name'] if top1 else '(无结果)',
                'hit': hit,
                'final': round(results[0]['final'], 4) if results else 0.0,
            }
            print(f'{query:<12}{label:<8}{dt:>10.1f}{"OK" if hit else "-":>6}  '
                  f'{rows[query][label]["top1"][:40]}')

        d = set(rows[query]['Dense']['ids'])
        s = set(rows[query]['Sparse']['ids'])
        rows[query]['overlap'] = round(len(d & s) / TOP_K * 100)

    # 汇总
    print('\n' + '=' * 88)
    print('【汇总】')
    print('=' * 88)
    for mode, label in MODES:
        hits = sum(rows[q][label]['hit'] for q in rows)
        avg_ms = sum(rows[q][label]['ms'] for q in rows) / len(rows)
        print(f'{label:<8} Top-1 命中 {hits}/{len(rows)}   平均耗时 {avg_ms:.1f} ms')

    print('\n【Dense 与 Sparse 的 Top-3 重叠度（按 id 计算）】')
    for query in rows:
        print(f'  {query:<12} {rows[query]["overlap"]:>3}%')

    print('\n测试完成！')


if __name__ == '__main__':
    main()
