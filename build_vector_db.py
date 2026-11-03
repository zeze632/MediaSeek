"""
构建 JSON 向量库：读取四个模态的标注数据，用 BGE-M3 编码，生成 JSON 文件

整体流程：
  Excel表格 → pandas读取 → 拼接文本 → BGE-M3编码 → JSON文件
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决 PyTorch 和 sklearn 的 OpenMP 冲突

import json
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

# ===== 配置路径 =====
BASE = '多模态数据/multi_data'  # 原始数据文件夹
OUTPUT_DIR = '向量库'            # 生成的 JSON 文件存放位置
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 如果输出文件夹不存在就创建一个

# ===== 1. 加载 BGE-M3 模型 =====
print('加载 BGE-M3 模型...')
# 从本地 bge-m3 文件夹加载 2.2GB 的模型文件
# use_fp16=True 表示用半精度，速度更快，精度微降
model = BGEM3FlagModel('bge-m3', use_fp16=True)
print('模型加载完成！\n')


def valid_file(file_name, folder):
    """文件名必须是字符串，且文件真的在磁盘上；否则这条标注就是死链接，跳过"""
    if not isinstance(file_name, str) or not file_name.strip():
        print(f'  跳过（文件名为空）: {file_name!r}')
        return False
    if not os.path.exists(os.path.join(folder, file_name)):
        print(f'  跳过（磁盘无此文件）: {file_name}')
        return False
    return True


def join_text(*values, fallback=''):
    """把若干标注字段拼成检索文本。

    空单元格在 pandas 里是 NaN，直接 f-string 会拼出字面量 "nan"
    （音频表里 30分钟纯音乐.mp3 就是这样变成 'nan' 的），所以统一在这里过滤，
    全空时退回文件名，保证每条素材都有可检索的文本。
    """
    parts = [str(v).strip() for v in values if not pd.isna(v) and str(v).strip()]
    return ' '.join(parts) if parts else fallback


def encode_and_build(records, db_description, output_filename):
    """
    通用函数：把一批数据编码成向量，生成 JSON 向量库文件

    参数:
        records: 一个列表，每项是一个字典，包含 id/file_name/type/dense_text/sparse_text
        db_description: 数据库描述文字
        output_filename: 输出 JSON 文件名（如 db_image.json）

    做的事:
        1. 把 records 里所有文本收集起来
        2. 一次性批量喂给 BGE-M3 编码（比逐条编码快）
        3. 把编码结果（dense + sparse 向量）填回每条数据里
        4. 整体写成 JSON 文件
    """
    print(f'正在处理: {output_filename}（共 {len(records)} 条）')

    # ---- 第1步：收集所有文本 ----
    # 从 records 里把每条的 dense_text 和 sparse_text 分别抽出来，组成两个列表
    dense_texts = [r['dense_text'] for r in records]    # 例如: ["图片描述1", "图片描述2", ...]
    sparse_texts = [r['sparse_text'] for r in records]  # 例如: ["图片描述1", "图片描述2", ...]

    # ---- 第2步：批量编码（dense + sparse 一次取回，同一批文本只算一遍）----
    #   - dense 编码：每段文本 → 1024 个小数（语义向量）
    #   - sparse 编码：每段文本 → {token编号: 权重}（关键词权重）
    print(f'  编码中...')
    output = model.encode(dense_texts, batch_size=8, max_length=512,
                          return_dense=True, return_sparse=True)

    # ---- 第3步：组装 JSON 结构 ----
    items = []
    for i, r in enumerate(records):
        # 每条数据就是一个字典，包含：
        #   - 原始信息：id, file_name, type, dense_text, sparse_text
        #   - 编码结果：encoding 里的 dense_vector 和 sparse_vector
        item = {
            "id": r['id'],                          # 唯一标识，如 "img_001"
            "file_name": r['file_name'],            # 原始文件名，如 "image1.png"
            "type": r.get('type', ''),              # 模态类型：image/video/audio/pdf
            "dense_text": r['dense_text'],          # 编码前的文本（语义搜索用）
            "sparse_text": r['sparse_text'],        # 编码前的文本（关键词搜索用）
            "encoding": {
                "model": "bge-m3",                  # 用的什么模型
                # dense_output['dense_vecs'][i] 是 numpy 数组，转成 Python 浮点数列表
                "dense_vector": [float(v) for v in output['dense_vecs'][i]],
                # output['lexical_weights'][i] 是 {token_id: float32} 字典
                # token_id 转成 Python int，权重转成 Python float
                "sparse_vector": {int(k): float(v) for k, v in output['lexical_weights'][i].items()}
            }
        }
        items.append(item)  # 把当前条目加入列表

    # 最外层结构：version + description + items 列表
    result = {
        "version": "1.0",                    # 数据库版本号
        "description": db_description,       # 数据库描述
        "items": items                       # 所有数据条目
    }

    # ---- 第4步：写入 JSON 文件 ----
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        # json.dump 把 Python 字典序列化为 JSON 字符串
        # ensure_ascii=False 保留中文不转义
        # indent=2 缩进 2 格，方便人阅读
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'  → 已保存: {output_path}\n')


# ===== 2. 处理图片数据 =====
# 图片 Excel 结构（格式不太规范，所以用列号而不是列名来取值）：
#   列0: file_name（文件名）
#   列5: type（类型，这里统一写 "image"）
#   列6: main_desc（主描述，一句话概括）
#   列9: sub_desc（详细描述，补充说明）
print('=' * 60)
print('【1/4】处理图片数据')
print('=' * 60)

# header=None 表示第一行也是数据，不当成列名
df_img = pd.read_excel(f'{BASE}/images/Image_annotation.xlsx', header=None)

records_img = []  # 用来存所有图片数据
IMG_DIR = f'{BASE}/images/all_images'
for idx, row in df_img.iterrows():  # 逐行读取

    # 按列号取数据（因为图片 Excel 列名不规范，用列号更可靠）
    file_name = row[0]          # 第 0 列：文件名，如 "image1.png"
    img_type = 'image'          # 固定写死为 "image"
    main_desc = row[6]          # 第 6 列：主描述
    sub_desc = row[9]           # 第 9 列：详细描述

    # 这张表没有表头（第 0 行就是 image2.jpg），所以不能按 idx==0 跳过，
    # 改为判断文件名是否像图片：表头 "file_name" 和空行都会被滤掉
    if not isinstance(file_name, str) or os.path.splitext(file_name)[1].lower() not in ('.jpg', '.jpeg', '.png', '.webp'):
        continue

    # 文件不在磁盘上就不入库，避免留下点开 404 的死条目
    if not os.path.exists(os.path.join(IMG_DIR, file_name)):
        print(f'  跳过（磁盘无此文件）: {file_name}')
        continue

    if pd.isna(main_desc):      # 如果主描述是空的，跳过这条
        continue

    # 拼接文本：主描述 + 详细描述，作为编码输入
    text = join_text(main_desc, sub_desc,
                     fallback=os.path.splitext(str(file_name))[0])

    # 存为字典，方便后面统一处理
    records_img.append({
        'id': f"img_{len(records_img)+1:03d}",  # 自动编号：img_001, img_002, ...
        'file_name': str(file_name),
        'type': str(img_type),
        'dense_text': text,    # 编码前的原文
        'sparse_text': text    # sparse 和 dense 用同样的文本
    })

# 统一编码 + 生成 JSON
encode_and_build(records_img, '图像向量数据库', 'db_image.json')


# ===== 3. 处理视频数据 =====
# 视频 Excel 结构（格式规范，能用列名取值）：
#   file_name: 视频文件名
#   type: 视频类型（这里统一写 "video"）
#   main_character: 主要人物/主题
#   main_desc: 主描述
#   sub_desc: 详细描述
print('=' * 60)
print('【2/4】处理视频数据')
print('=' * 60)

df_video = pd.read_excel(f'{BASE}/videos/video_annotation.xlsx')  # 有列名，不用 header=None

records_video = []
for idx, row in df_video.iterrows():
    file_name = row['file_name']              # 用列名取值
    if not valid_file(file_name, f'{BASE}/videos/all_video'):
        continue
    video_type = 'video'                       # 固定写死为 "video"
    # main_character 可能是空单元格（上传的视频没有这一列），要当成空串处理，
    # 否则 f-string 会把 NaN 拼成 "nan: 描述"
    main_character = '' if pd.isna(row['main_character']) else str(row['main_character']).strip()
    main_desc = row['main_desc']               # 如 "《告台湾同胞书》..."
    sub_desc = row['sub_desc'] if not pd.isna(row['sub_desc']) else ''  # 详细描述可能为空

    # 视频的文本：人物 + 主描述 + 详细描述，拼在一起
    prefix_text = f'{main_character}:' if main_character else ''
    text = join_text(prefix_text, main_desc, sub_desc,
                     fallback=os.path.splitext(str(file_name))[0])

    records_video.append({
        'id': f"video_{len(records_video)+1:03d}",
        'file_name': str(file_name),
        'type': str(video_type),
        'dense_text': text,
        'sparse_text': text
    })

encode_and_build(records_video, '视频向量数据库', 'db_video.json')


# ===== 4. 处理音频数据 =====
# 音频 Excel 结构：
#   file_name: 音频文件名
#   type: 音频类型（这里统一写 "audio"）
#   main_desc: 主描述
#   sub_desc: 详细描述
print('=' * 60)
print('【3/4】处理音频数据')
print('=' * 60)

df_audio = pd.read_excel(f'{BASE}/audios/audio-annotation.xlsx')

records_audio = []
for idx, row in df_audio.iterrows():
    file_name = row['file_name']
    if not valid_file(file_name, f'{BASE}/audios/all_audios'):
        continue
    audio_type = 'audio'                       # 固定写死为 "audio"
    main_desc = row['main_desc']
    sub_desc = row['sub_desc'] if not pd.isna(row['sub_desc']) else ''
    # 音频文本：主描述 + 详细描述；两者都空时退回文件名（如 30分钟纯音乐.mp3）
    text = join_text(main_desc, sub_desc,
                     fallback=os.path.splitext(str(file_name))[0])

    records_audio.append({
        'id': f"audio_{len(records_audio)+1:03d}",
        'file_name': str(file_name),
        'type': str(audio_type),
        'dense_text': text,
        'sparse_text': text
    })

encode_and_build(records_audio, '音频向量数据库', 'db_audio.json')


# ===== 5. 处理 PDF 数据 =====
# PDF Excel 结构（最简单，只有两列）：
#   file_name: PDF 文件名
#   title: 政策标题（就直接用标题当检索文本）
print('=' * 60)
print('【4/4】处理 PDF 数据')
print('=' * 60)

df_pdf = pd.read_excel(f'{BASE}/pdf/pdf_annotation.xlsx')

records_pdf = []
for idx, row in df_pdf.iterrows():
    file_name = row['file_name']
    if not valid_file(file_name, f'{BASE}/pdf/all_pdf'):
        continue
    title = row['title']  # 如 "广东省人民政府关于..."
    text = join_text(title, fallback=os.path.splitext(str(file_name))[0])

    records_pdf.append({
        'id': f"pdf_{len(records_pdf)+1:03d}",
        'file_name': str(file_name),
        'type': 'pdf',                    # 固定写死为 "pdf"
        'dense_text': text,               # PDF 只用标题当检索文本
        'sparse_text': text
    })

encode_and_build(records_pdf, 'PDF文档向量数据库', 'db_pdf.json')


# ===== 6. 汇总 =====
print('=' * 60)
print('全部完成！')
print(f'图片: {len(records_img)} 条')
print(f'视频: {len(records_video)} 条')
print(f'音频: {len(records_audio)} 条')
print(f'PDF:  {len(records_pdf)} 条')
print(f'总计: {len(records_img)+len(records_video)+len(records_audio)+len(records_pdf)} 条')
print(f'输出目录: {OUTPUT_DIR}/')
