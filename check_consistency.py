# -*- coding: utf-8 -*-
"""
数据一致性检查：Excel 标注 ↔ 磁盘文件 ↔ JSON 向量库 ↔ ChromaDB

跑法：python check_consistency.py
任何一处不一致都会列出来，退出码非 0，方便放进流程里当检查关卡。
"""
import os
import json
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import chromadb

BASE = '多模态数据/multi_data'

MODALITIES = {
    'image': {
        'xlsx': f'{BASE}/images/Image_annotation.xlsx',
        'folder': f'{BASE}/images/all_images',
        'header': None,
    },
    'video': {
        'xlsx': f'{BASE}/videos/video_annotation.xlsx',
        'folder': f'{BASE}/videos/all_video',
        'header': 0,
    },
    'audio': {
        'xlsx': f'{BASE}/audios/audio-annotation.xlsx',
        'folder': f'{BASE}/audios/all_audios',
        'header': 0,
    },
    'pdf': {
        'xlsx': f'{BASE}/pdf/pdf_annotation.xlsx',
        'folder': f'{BASE}/pdf/all_pdf',
        'header': 0,
    },
}

MEDIA_EXT = {'.jpg', '.jpeg', '.png', '.webp', '.mp4', '.mov', '.avi', '.mkv',
             '.mp3', '.wav', '.m4a', '.flac', '.pdf'}


def excel_names(cfg):
    df = pd.read_excel(cfg['xlsx'], header=cfg['header'])
    col = 0 if cfg['header'] is None else 'file_name'
    return [str(v) for v in df[col].tolist()]


def disk_files(folder):
    return {f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in MEDIA_EXT}


def json_names(tag):
    with open(f'向量库/db_{tag}.json', encoding='utf-8') as f:
        return [i['file_name'] for i in json.load(f)['items']]


def chroma_counts():
    client = chromadb.PersistentClient(path='./chroma_db')
    result = {}
    for name in ['multi_modal', 'doc_chunks']:
        try:
            col = client.get_collection(name)
            metas = col.get(include=['metadatas'])['metadatas']
            result[name] = metas
        except Exception as e:
            print(f'  [警告] 读取集合 {name} 失败: {e}')
    return result


def main():
    problems = []
    print('=' * 72)
    print(f'{"模态":<8}{"Excel":>8}{"磁盘":>8}{"JSON":>8}{"Chroma":>8}   状态')
    print('=' * 72)

    chroma_meta = chroma_counts()
    multi = chroma_meta.get('multi_modal', [])
    chroma_by_type = {}
    for m in multi:
        chroma_by_type.setdefault(m.get('db_type'), set()).add(m.get('file_name'))

    for tag, cfg in MODALITIES.items():
        ex = excel_names(cfg)
        dk = disk_files(cfg['folder'])
        js = json_names(tag)
        ch = chroma_by_type.get(tag, set())

        ok = (len(ex) == len(dk) == len(js) == len(ch)
              and set(ex) == set(dk) == set(js) == ch)
        print(f'{tag:<8}{len(ex):>8}{len(dk):>8}{len(js):>8}{len(ch):>8}   {"OK" if ok else "不一致"}')

        # 逐项列出差异，方便定位
        for label, got, want in [
            ('Excel 与磁盘', set(ex), dk),
            ('JSON 与磁盘', set(js), dk),
            ('Chroma 与磁盘', ch, dk),
        ]:
            missing = sorted(want - got)     # 磁盘有、库里没有
            extra = sorted(got - want)       # 库里有、磁盘没有
            if missing:
                problems.append(f'{tag}: {label} 缺 {len(missing)} 个 -> {missing[:5]}')
            if extra:
                problems.append(f'{tag}: {label} 多 {len(extra)} 个（磁盘无此文件）-> {extra[:5]}')

        dup = {n for n in ex if ex.count(n) > 1}
        if dup:
            problems.append(f'{tag}: Excel 有重复文件名 -> {sorted(dup)}')

    # 总量
    total_json = sum(len(json_names(t)) for t in MODALITIES)
    total_chroma = len(multi)
    print('-' * 72)
    print(f'JSON 总条数: {total_json} | Chroma multi_modal: {total_chroma} '
          f'| doc_chunks: {len(chroma_meta.get("doc_chunks", []))}')

    # ID 唯一性
    all_ids = []
    for t in MODALITIES:
        with open(f'向量库/db_{t}.json', encoding='utf-8') as f:
            all_ids += [i['id'] for i in json.load(f)['items']]
    if len(all_ids) != len(set(all_ids)):
        problems.append(f'JSON 里存在重复 ID（{len(all_ids)} 条 / {len(set(all_ids))} 个唯一）')

    print('=' * 72)
    if problems:
        print(f'发现 {len(problems)} 个问题：')
        for p in problems:
            print('  -', p)
        sys.exit(1)
    else:
        print('全部一致，没有发现问题。')


if __name__ == '__main__':
    main()
