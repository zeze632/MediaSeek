# -*- coding: utf-8 -*-
"""
检索组件：BM25 关键词检索 + RRF 排名融合

为什么单独放一个文件：
- BM25 的数学和网页接口无关，拆出来能单独测
- web_app.py 已经 480 行，再塞进去会很难读
"""
import math
from collections import Counter

import jieba


def tokenize(text):
    """中文分词，统一小写，丢掉纯空白"""
    return [w.strip().lower() for w in jieba.lcut(str(text)) if w.strip()]


# 查询侧的停用词：这些词在问句里几乎不携带信息，却会被 BM25 当成关键词，
# 把“什么、哪些、怎么”命中的无关文档排到前面，所以要滤掉
QUERY_STOPWORDS = set("""
的 了 是 在 和 与 及 等 中 对 为 从 到 按 向 将 以 被 把 着 之 其 不 也 都
就 要 会 能 可 有 这 那 上 下 里 外 个 些 过 而 但 或 我 你 他 她 它 吗 呢 吧
啊 什么 哪些 哪个 哪里 怎么 怎样 如何 为什么 多少 是否 请问 一下 一些 这些
那些 这个 那个 我们 你们 他们 内容 情况 问题 时候 可以 需要 进行 关于 讲了
说说 谈谈 介绍 有关 相关 以及 其中 目前 现在
""".split())


def tokenize_query(text):
    """查询专用分词：去掉停用词和单字，避免噪音进 BM25"""
    return [w for w in tokenize(text)
            if len(w) >= 2 and w not in QUERY_STOPWORDS]


class BM25Index:
    """Okapi BM25

    k1: 词频饱和速度，越大越慢饱和（1.2~2.0 是常用区间）
    b : 文档长度惩罚强度，0 完全不惩罚，1 完全按长度归一
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n = len(self.doc_tokens)
        self.avg_len = sum(self.doc_len) / max(self.n, 1)
        self.tf = [Counter(t) for t in self.doc_tokens]

        # 文档频率：某个词出现在多少篇文档里
        df = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        # BM25 的 IDF：稀有词权重高，常见词接近 0
        self.idf = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5))
            for w, c in df.items()
        }

    def scores(self, query_tokens):
        """返回与文档顺序一一对应的 BM25 得分"""
        out = [0.0] * self.n
        for w in query_tokens:
            idf = self.idf.get(w)
            if idf is None:      # 词表里没有的词直接跳过
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                out[i] += idf * f * (self.k1 + 1) / denom
        return out


def rrf_fuse(rankings, k=60, weights=None):
    """Reciprocal Rank Fusion

    不看分数，只看名次：第 r 名贡献 w/(k+r)。
    多路结果里都排得靠前的文档，融合分自然最高。
    k 一般取 60，作用是压平头部差距。
    weights 用来给不同路降权（关键词路通常配比语义路低）。
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores = {}
    for weight, ranking in zip(weights, rankings):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
