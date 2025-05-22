import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class FontSampler:
    def __init__(self, fonts_dir, text_file, chars_file, max_fonts=None, font_size=76):
        """
        初始化 FontSampler 类。

        :param fonts_dir: 字体文件夹路径
        :param text_file: 文本文件路径
        :param chars_file: 常用字文件路径
        :param max_fonts: 最大加载的字体数量
        :param font_size: 字体大小
        """
        self.font_files = self.load_fonts(fonts_dir, max_fonts)
        self.text = self.load_text(text_file)
        self.chars = self.load_text(chars_file).strip()
        self.image_maps = self.generate_image_map(self.font_files, self.chars, font_size)

    @staticmethod
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

    @staticmethod
    def load_text(text_file):
        """
        加载文本文件内容。

        :param text_file: 文本文件路径
        :return: 文本内容
        """
        with open(text_file, 'r', encoding='utf-8') as file:
            text_content = file.read()
        return text_content

    @staticmethod
    def render_character(font, character, image_size=84):
        """
        将字符渲染为指定大小的灰度图像。
        先在 128x128 的画布上绘制字符，然后使用 NumPy 计算实际边界并裁剪到 84x84。

        :param font: 字体对象
        :param character: 要渲染的字符
        :param image_size: 最终裁剪的图像大小
        :return: 渲染后的图像
        """
        canvas_size = 128
        canvas = Image.new("L", (canvas_size, canvas_size), color=255)
        draw = ImageDraw.Draw(canvas)
        text_offset = (16, 8)
        draw.text(text_offset, character, font=font, fill=0)
        image_array = np.array(canvas)
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
        char_center_x = (left + right) / 2
        char_center_y = (top + bottom) / 2
        crop_left = max(0, char_center_x - (image_size / 2))
        crop_top = max(0, char_center_y - (image_size / 2))
        crop_right = min(canvas_size, crop_left + image_size)
        crop_bottom = min(canvas_size, crop_top + image_size)
        cropped_image = canvas.crop((crop_left, crop_top, crop_right, crop_bottom))
        return cropped_image

    def generate_image_map(self, font_files, chars, font_size=72):
        """
        为每个字体生成图像 map。

        :param font_files: 字体文件路径列表
        :param chars: 常用字列表
        :param font_size: 字体大小
        :return: 图像map列表，每个图像map对应一个字体
        """
        image_maps = []
        for font_path in tqdm(font_files, desc="Loading fonts and rendering characters"):
            font = ImageFont.truetype(font_path, font_size)
            image_map = {}
            for char in chars:
                try:
                    image = self.render_character(font, char)
                    image_map[char] = image
                except Exception as e:
                    print(f"Error rendering character '{char}' in font {font_path}: {e}")
            image_maps.append(image_map)
        return image_maps

    import random
    from PIL import Image, ImageDraw, ImageFont

    def sample(self, sample_cnt, rotation_range=(-4, 4), translation_range=(-2, 2), scale_range=(0.92, 1.0)):
        """
        随机采样：从每个字体的图像map中生成样本。
        每个样本包括一张图片和字体编号。
        图片大小为 224x224，包括 5x5=25 个字
        从 text 的内容中随机取连续的 25 个字，尝试从左到右依次填充到图片中。
        如果遇到常用字中不包含的字，则重新随机。

        :param sample_cnt: 每个字体的采样次数
        :param rotation_range: 随机旋转范围
        :param translation_range: 随机平移范围
        :param scale_range: 随机缩放范围
        :return: 样本列表
        """
        samples = []
        text_length = len(self.text)
        char_set = set(self.chars)

        for font_id, font_map in enumerate(self.image_maps):
            for _ in range(sample_cnt):
                while True:
                    start_idx = random.randint(0, text_length - 25)
                    selected_chars = self.text[start_idx:start_idx + 25]
                    if all(char in char_set for char in selected_chars):
                        break

                image = Image.new("L", (224, 224), color=255)
                char_size = 42
                spacing = 2

                for i, char in enumerate(selected_chars):
                    x = spacing + (i % 5) * (char_size + spacing)
                    y = spacing + (i // 5) * (char_size + spacing)

                    char_image = font_map.get(char)
                    if char_image:
                        # 先对 84x84 的原图进行旋转和缩放
                        if rotation_range:
                            angle = random.uniform(rotation_range[0], rotation_range[1])
                            char_image = char_image.rotate(angle, expand=True, fillcolor=255)

                        if scale_range:
                            scale = random.uniform(scale_range[0], scale_range[1])
                            new_size = (int(84 * scale), int(84 * scale))
                            char_image = char_image.resize(new_size, Image.LANCZOS)

                        if translation_range:
                            dx = random.uniform(translation_range[0], translation_range[1])
                            dy = random.uniform(translation_range[0], translation_range[1])
                            x += round(dx)
                            y += round(dy)

                        # 调整字符图像的大小为 42x42
                        char_image = char_image.resize((char_size, char_size), Image.LANCZOS)

                        # 将字符图像粘贴到目标图像上
                        image.paste(char_image, (x, y))

                samples.append((image, font_id))

        return samples

    @staticmethod
    def save_samples(samples, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx, (image, font_id) in enumerate(samples):
            output_path = os.path.join(output_dir, f"sample_{idx}_font_{font_id}.png")
            image.save(output_path)

# 测试代码
if __name__ == "__main__":
    fonts_dir = "fonts"  # 字体文件夹路径
    text_file = "text.txt"  # 文本文件路径
    chars_file = "chars.txt"  # 常用字文件路径
    output_dir = "sample"  # 输出文件夹路径

    sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=16, font_size=76)
    samples = sampler.sample(sample_cnt=8)
    sampler.save_samples(samples, output_dir)