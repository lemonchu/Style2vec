import os
from PIL import Image, ImageOps
import numpy as np
import cv2
from tqdm import tqdm

raw_data_dir = 'candidates'
scaled_data_dir = 'scaled_candidates'
os.makedirs(scaled_data_dir, exist_ok=True)

def compute_sample_ratio(np_img, grid=5, threshold=128):
    """
    对输入正方形二值图（np.uint8）划分为 grid x grid 个单元，
    对每个单元计算其中“字”的最小包含方框，然后计算该包含方框的长边占单元格边长的比例，
    最后返回该样本所有单元比例的平均值。

    如果该单元内无黑点（<threshold），则按单元尺寸返回比例1.0。
    """
    h, w = np_img.shape
    cell_size = w // grid
    ratios = []
    for i in range(grid):
        for j in range(grid):
            left = j * cell_size
            upper = i * cell_size
            cell = np_img[upper:upper+cell_size, left:left+cell_size]
            # 找到黑色区域（设定阈值，像素 < threshold 认为是笔画）
            coords = np.argwhere(cell < threshold)
            if coords.size == 0:
                # 若没有黑点，则认为字符占整个单元
                ratio = 1.0
                bbox = (0, 0, cell_size, cell_size)
            else:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                w_box = x1 - x0 + 1
                h_box = y1 - y0 + 1
                long_edge = max(w_box, h_box)
                ratio = long_edge / cell_size
            ratios.append(ratio)
    sample_avg_ratio = np.mean(ratios)
    return sample_avg_ratio

import cv2

def process_sample_image(img, global_avg_ratio, grid=5, threshold=128):
    """
    对已裁剪为正方形（binary图）的 PIL Image 进行如下处理：
    1. 将图像均匀划分为 grid x grid 个单元；
    2. 对每个单元内，通过连通块检测判断是否存在部分来自其他字的噪声，
       如果连通块的重心位于单元外侧 15% 边缘区域，则将其剔除（设为白色）；
    3. 计算当前字的 bounding box（黑色区域）的长边占单元格边长比例，进而计算样本平均比例 sample_ratio；
    4. 计算缩放因子 factor = global_avg_ratio / sample_ratio；
    5. 按 factor 对每个单元内的字符图像缩放，并将缩放后图像居中放入固定尺寸单元中；
    6. 拼接所有单元后调整为目标大小（例如 224x224）。
    """
    np_img = np.array(img, dtype=np.uint8)
    h, w = np_img.shape
    cell_size = w // grid

    # 计算当前样本的平均比例（对每个单元采用原始二值图计算）
    sample_ratio = compute_sample_ratio(np_img, grid, threshold)
    factor = global_avg_ratio / sample_ratio

    new_sample = Image.new("L", (cell_size * grid, cell_size * grid), color=255)

    for i in range(grid):
        for j in range(grid):
            left = j * cell_size
            upper = i * cell_size
            cell = img.crop((left, upper, left + cell_size, upper + cell_size))
            cell_np = np.array(cell, dtype=np.uint8)
            # 生成二值 mask：黑色（笔画）为1，背景为0
            mask = (cell_np < threshold).astype(np.uint8)
            # 连通块检测，注意背景为0，连通块标签从1开始
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            # 对每个连通块检查其重心是否在外侧 15% 区域内
            h_cell, w_cell = mask.shape
            for comp in range(1, num_labels):
                cx, cy = centroids[comp]
                if (cx < 0.05 * w_cell or cx > 0.95 * w_cell or 
                    cy < 0.05 * h_cell or cy > 0.95 * h_cell):
                    # 剔除此连通块：将对应区域置为白色（255）
                    cell_np[labels == comp] = 255
            # 根据剔除之后的 cell_np 重新计算 bounding box
            coords = np.argwhere(cell_np < threshold)
            if coords.size != 0:
                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)
                # 可追加额外边界扩展（如需扩展，可在此处调整，例如扩展4像素，但确保不超出 cell 范围）
                x0_exp = max(0, x0 - 4)
                y0_exp = max(0, y0 - 4)
                x1_exp = min(w_cell, x1 + 4)
                y1_exp = min(h_cell, y1 + 4)
                bbox = (x0_exp, y0_exp, x1_exp, y1_exp)
                char_img = Image.fromarray(cell_np).crop(bbox)
            else:
                char_img = Image.fromarray(cell_np).copy()
            # 调整字符大小
            new_w = max(1, int(char_img.width * factor))
            new_h = max(1, int(char_img.height * factor))
            char_img_resized = char_img.resize((new_w, new_h), Image.LANCZOS)
            # 将缩放后图像居中放入固定尺寸单元中
            cell_new = Image.new("L", (cell_size, cell_size), color=255)
            paste_x = (cell_size - new_w) // 2
            paste_y = (cell_size - new_h) // 2
            cell_new.paste(char_img_resized, (paste_x, paste_y))
            new_sample.paste(cell_new, (j * cell_size, i * cell_size))
    final_sample = new_sample.resize((224, 224), Image.LANCZOS)
    return final_sample

# 第一遍：遍历所有图片，计算每个样本的字长比例平均值
sample_ratios = []
filenames_first = []
for filename in tqdm(os.listdir(raw_data_dir), desc="First pass: compute ratios"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(raw_data_dir, filename)
        try:
            with Image.open(input_path) as img:
                img_gray = img.convert('L')
                # 二值化
                binary = img_gray.point(lambda p: 255 if p > 128 else 0)
                # 裁剪为正方形（居中裁剪）
                width, height = binary.size
                if width > height:
                    left = (width - height) // 2
                    upper = 0
                    right = left + height
                    lower = height
                else:
                    left = 0
                    upper = (height - width) // 2
                    right = width
                    lower = upper + width
                square_img = binary.crop((left, upper, right, lower))
                # 计算本样本的平均比例
                ratio = compute_sample_ratio(np.array(square_img, dtype=np.uint8))
                sample_ratios.append(ratio)
                filenames_first.append(filename)
        except Exception as e:
            print(f"Failed to process {filename} in first pass: {e}")

if not sample_ratios:
    print("No valid stroke ratios computed!")
    exit(1)

global_avg_ratio = np.mean(sample_ratios)
print(f"Global average ratio: {global_avg_ratio:.4f}")

# 第二遍：根据 global_avg_ratio 对每个样本做缩放及字居中处理，然后保存
for filename in tqdm(os.listdir(raw_data_dir), desc="Second pass: adjust and save"):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(raw_data_dir, filename)
        try:
            with Image.open(input_path) as img:
                img_gray = img.convert('L')
                binary = img_gray.point(lambda p: 255 if p > 128 else 0)
                # 裁剪为正方形
                width, height = binary.size
                if width > height:
                    left = (width - height) // 2
                    upper = 0
                    right = left + height
                    lower = height
                else:
                    left = 0
                    upper = (height - width) // 2
                    right = width
                    lower = upper + width
                square_img = binary.crop((left, upper, right, lower))
                # 根据 global_avg_ratio 调整每个样本
                final_sample = process_sample_image(square_img, global_avg_ratio, grid=5, threshold=128)
                # 保存输出文件（统一保存为 png）
                output_path = os.path.join(scaled_data_dir, os.path.splitext(filename)[0] + '.png')
                final_sample.save(output_path)
        except Exception as e:
            print(f"Failed to process {filename} in second pass: {e}")

print(f"All images processed and saved to {scaled_data_dir}.")