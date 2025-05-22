import os
import random
from tqdm import tqdm  # 导入 tqdm 用于显示进度条

# 加载字体文件
def load_fonts(fonts_dir, max_fonts=None):
    """
    加载所有 ttf 字体文件。

    :param fonts_dir: 字体文件夹路径
    :param max_fonts: 最大加载的字体数量
    :return: 字体路径列表
    """
    font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith('.ttf')]
    if max_fonts is not None:
        font_files = font_files[:max_fonts]
    return font_files

# 加载文本内容
def load_text(text_file):
    """
    加载文本文件内容。

    :param text_file: 文本文件路径
    :return: 文本内容
    """
    with open(text_file, 'r', encoding='utf-8') as file:
        text_content = file.read()
    return text_content

# 渲染字符为图像
from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

from PIL import Image, ImageDraw, ImageFont

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def render_character(font, character, image_size=84):
    """
    将字符渲染为指定大小的灰度图像。
    先在 128x128 的画布上绘制字符，然后使用 NumPy 计算实际边界并裁剪到 84x84。

    :param font: 字体对象
    :param character: 要渲染的字符
    :param image_size: 最终裁剪的图像大小
    :return: 渲染后的图像
    """
    # 创建 128x128 的画布
    canvas_size = 128
    canvas = Image.new("L", (canvas_size, canvas_size), color=255)
    draw = ImageDraw.Draw(canvas)

    # 在画布上以 (16, 8) 的偏移量绘制字符
    text_offset = (16, 8)
    draw.text(text_offset, character, font=font, fill=0)

    # 将图像转换为 NumPy 数组
    image_array = np.array(canvas)

    # 找到非零像素的边界
    rows = np.any(image_array < 128, axis=1)
    cols = np.any(image_array < 128, axis=0)
    if not np.any(rows) or not np.any(cols):
        top = left = 0
        bottom = right = 127
    else:
        top = np.argmax(rows)
        bottom = len(rows) - np.argmax(rows[::-1]) - 1
        left = np.argmax(cols)
        right = len(cols) - np.argmax(cols[::-1]) - 1

    # 计算字符的实际中心
    char_center_x = (left + right) / 2
    char_center_y = (top + bottom) / 2

    # 计算裁剪框的左上角和右下角坐标
    crop_left = char_center_x - (image_size / 2)
    crop_top = char_center_y - (image_size / 2)
    crop_right = crop_left + image_size
    crop_bottom = crop_top + image_size

    # 确保裁剪框不超出画布范围
    crop_left = max(0, crop_left)
    crop_top = max(0, crop_top)
    crop_right = min(canvas_size, crop_right)
    crop_bottom = min(canvas_size, crop_bottom)

    # 裁剪到 84x84
    cropped_image = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))

    return cropped_image

# 生成图像map
def generate_image_map(font_files, common_chars, font_size=72):
    """
    为每个字体生成图像 map。

    :param font_files: 字体文件路径列表
    :param common_chars: 常用字列表
    :param font_size: 字体大小
    :return: 图像map列表，每个图像map对应一个字体
    """
    image_maps = []
    for font_path in tqdm(font_files, desc="Loading fonts and rendering characters"):
        font = ImageFont.truetype(font_path, font_size)
        image_map = {}
        for char in common_chars:
            try:
                image = render_character(font, char)
                image_map[char] = image
            except Exception as e:
                print(f"Error rendering character '{char}' in font {font_path}: {e}")
        image_maps.append(image_map)
    return image_maps

def random_sample(fonts, text, chars, sample_cnt, rotation_range=(-4,4), translation_range=(-2,2), scale_range=(0.92,1.0)):
    """
    随机采样：从每个字体的图像map中生成样本。
    每个样本包括一张图片和字体编号。
    图片大小为 224x224，包括 5x5=25 个字
    从 text 的内容中随机取连续的 25 个字，尝试从左到右依次填充到图片中。
    如果遇到常用字中不包含的字，则重新随机。
    """

    samples = []
    text_length = len(text)
    char_set = set(chars)  # 常用字集合，用于快速查找

    for font_id, font_map in enumerate(fonts):
        for _ in range(sample_cnt):
            while True:
                # 随机选择连续的 25 个字
                start_idx = random.randint(0, text_length - 25)
                selected_chars = text[start_idx:start_idx + 25]

                # 检查是否所有字符都在常用字中
                if all(char in char_set for char in selected_chars):
                    break

            # 创建 224x224 的图片
            image = Image.new("L", (224, 224), color=255)
            draw = ImageDraw.Draw(image)

            # 每个字的大小为 40x40，间距为 2
            char_size = 42
            spacing = 2

            # 将字符填充到图片中
            for i, char in enumerate(selected_chars):
                x = spacing + (i % 5) * (char_size + spacing)
                y = spacing + (i // 5) * (char_size + spacing)

                # 获取字符对应的图像（80x80），并缩小到 40x40
                char_image = font_map.get(char)
                if char_image:
                    char_image = char_image.resize((char_size, char_size), Image.LANCZOS)  # 使用线性插值缩小

                    # if rotation_range:
                    #     angle = random.uniform(rotation_range[0], rotation_range[1])
                    #     char_image = char_image.rotate(angle, expand=True, fillcolor=255)
                    #
                    # # 随机平移
                    # if translation_range:
                    #     dx = random.uniform(translation_range[0], translation_range[1])
                    #     dy = random.uniform(translation_range[0], translation_range[1])
                    #     x += round(dx)
                    #     y += round(dy)
                    #
                    # # 随机缩放
                    # if scale_range:
                    #     scale = random.uniform(scale_range[0], scale_range[1])
                    #     new_size = (int(char_size * scale), int(char_size * scale))
                    #     char_image = char_image.resize(new_size, Image.LANCZOS)

                    image.paste(char_image, (x, y))

            samples.append((image, font_id))

    return samples

def save_samples(samples, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, (image, font_id) in enumerate(samples):
        output_path = os.path.join(output_dir, f"sample_{idx}_font_{font_id}.png")
        image.save(output_path)

# 测试代码
if __name__ == "__main__":
    fonts_dir = "fonts"  # 字体文件夹路径
    output_dir = "sample"  # 输出文件夹路径

    font_files = load_fonts(fonts_dir, max_fonts=16)  # 最多加载5个字体
    text = load_text("text.txt")  # 加载文本文件
    chars = load_text("chars.txt")

    image_maps = generate_image_map(font_files, chars)  # 生成图像map
    samples = random_sample(image_maps, text, chars, sample_cnt=8)  # 生成样本
    save_samples(samples, output_dir)  # 保存样本图片