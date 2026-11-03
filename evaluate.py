# -*- coding: utf-8 -*-
"""
检索效果评测：对比 语义 / 关键词 / 混合(RRF) 三种方式

标注规则（写在 eval_set.json 里）：
  描述文本包含指定关键词的条目，算作该查询的「正确答案」。
  这是自动标注，好处是可复现；代价是偏向关键词检索，
  所以看结果时重点看三者的相对差距，而不是绝对值。

跑法：python evaluate.py
"""
import json
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from web_app import retrieve, all_items

TOP_K = 5

REFUSE_QUERIES = ['广彩瓷', '量子纠缠', '马拉松训练', '红烧肉做法',
                  '火星殖民', '股票期权', '芭蕾舞', '咖啡因']


def load_eval():
    with open('eval_set.json', encoding='utf-8') as f:
        return json.load(f)


def relevant_ids(terms):
    return {it['id'] for it in all_items
            if any(t in it['dense_text'] for t in terms)}


def evaluate(mode, cases):
    rows = []
    for case in cases:
        rel = relevant_ids(case['terms'])
        if not rel:
            continue
        results, _ = retrieve(case['query'], top_k=TOP_K, mode=mode)
        ids = [r['item']['id'] for r in results]

        hits = [i for i in ids if i in rel]
        recall = len(hits) / min(TOP_K, len(rel))

        rr = 0.0
        for rank, i in enumerate(ids, start=1):
            if i in rel:
                rr = 1.0 / rank
                break
        rows.append((case['query'], recall, rr, len(rel)))
    return rows


def main():
    cases = load_eval()
    modes = [('dense', '语义'), ('sparse', '关键词'), ('hybrid', '混合 RRF')]
    all_rows = {}

    print('=' * 78)
    print(f'{"查询":<14}' + ''.join(f'{label+" R@5":>13}{label+" MRR":>13}' for _, label in modes))
    print('=' * 78)

    for i, case in enumerate(cases):
        line = f'{case["query"]:<14}'
        for mode, _label in modes:
            rows = evaluate(mode, [case])
            if rows:
                _, recall, rr, _ = rows[0]
                all_rows.setdefault(mode, []).append((recall, rr))
                line += f'{recall:>13.2f}{rr:>13.2f}'
            else:
                line += f'{"-":>13}{"-":>13}'
        print(line)

    print('-' * 78)
    line = f'{"平均":<14}'
    summary = {}
    for mode, label in modes:
        vals = all_rows.get(mode, [])
        if vals:
            r = sum(v[0] for v in vals) / len(vals)
            m = sum(v[1] for v in vals) / len(vals)
            summary[mode] = (r, m)
            line += f'{r:>13.3f}{m:>13.3f}'
        else:
            line += f'{"-":>13}{"-":>13}'
    print(line)
    print('=' * 78)

    # 拒答能力：语料外的问题应当返回 0 条
    print('\n拒答测试（语料外问题，期望全部 0 条）：')
    ok = 0
    for q in REFUSE_QUERIES:
        results, _ = retrieve(q, top_k=TOP_K, mode='hybrid')
        n = len(results)
        ok += (n == 0)
        print(f'  {q:<10} 返回 {n} 条')
    print(f'  通过 {ok}/{len(REFUSE_QUERIES)}')

    return summary


if __name__ == '__main__':
    main()
