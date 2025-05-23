import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class FontSampler:
    def __init__(self, fonts_dir, text_file, chars_file, max_fonts=None, font_size=76, train_ratio=0.8):
        """
        初始化 FontSampler 类。

        :param fonts_dir: 字体文件夹路径
        :param text_file: 文本文件路径
        :param chars_file: 常用字文件路径
        :param max_fonts: 最大加载的字体数量
        :param font_size: 字体大小
        :param train_ratio: 用于训练的字体比例，其余用于测试
        """
        self.font_files = self.load_fonts(fonts_dir, max_fonts)
        self.train_font_files, self.test_font_files = self.split_fonts(self.font_files, train_ratio)
        self.text = self.load_text(text_file)
        self.chars = self.load_text(chars_file).strip()
        self.chars_set = set(self.chars)
        self.train_image_maps = self.generate_image_map(self.train_font_files, self.chars, font_size)
        self.test_image_maps = self.generate_image_map(self.test_font_files, self.chars, font_size)

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
        
        # 将字体随机打乱
        random.shuffle(font_files)
        
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

    def split_fonts(self, font_files, train_ratio):
        """
        将字体列表按照 train_ratio 分为训练和测试两部分。
        
        :param font_files: 字体文件路径列表
        :param train_ratio: 训练字体所占比例
        :return: (训练字体列表, 测试字体列表)
        """
        random.shuffle(font_files)
        split_index = int(len(font_files) * train_ratio)
        return font_files[:split_index], font_files[split_index:]

    def generate_image_map(self, font_files, chars, font_size=72):
        """
        为每个字体生成图像 map，使用多线程加速字符渲染。

        :param font_files: 字体文件路径列表
        :param chars: 常用字列表
        :param font_size: 字体大小
        :return: 图像map列表，每个图像map对应一个字体
        """
        image_maps = []

        def render_characters_for_font(font_path):
            """
            渲染单个字体的所有字符图像 map。
            """
            font = ImageFont.truetype(font_path, font_size)
            image_map = {}

            # 使用多线程渲染字符
            def render_single_character(char):
                return char, self.render_character(font, char)

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(render_single_character, char): char for char in chars}
                for future in futures:
                    char, image = future.result()
                    if image is not None:
                        image_map[char] = image

            return image_map

        # 渲染所有字体
        for font_path in tqdm(font_files, desc="Loading fonts and rendering characters"):
            image_map = render_characters_for_font(font_path)
            image_maps.append(image_map)

        return image_maps

    def sample(self, font_cnt, sample_cnt, rotation_range=(-4, 4), translation_range=(-2, 2), scale_range=(0.92, 1.0),
               sample_source="train"):
        """
        随机采样：从随机选择的字体子集中生成样本。
        每个样本包括一张图片和字体编号。
        图片大小为 224x224，包括 5x5=25 个字。
        从 text 的内容中随机取连续的 25 个字，尝试从左到右依次填充到图片中。
        如果遇到常用字中不包含的字，则重新随机。

        :param font_cnt: 随机选择的字体数量
        :param sample_cnt: 每个字体的采样次数
        :param rotation_range: 随机旋转范围
        :param translation_range: 随机平移范围
        :param scale_range: 随机缩放范围
        :param sample_source: 指定采样字体的来源，"train" 或 "test"
        :return: 样本列表
        """
        samples = []
        text_length = len(self.text)

        # 根据 sample_source 选择对应的字体图像map
        if sample_source == "train":
            image_maps = self.train_image_maps
        elif sample_source == "test":
            image_maps = self.test_image_maps
        else:
            raise ValueError("sample_source 必须为 'train' 或 'test'")

        # 随机选择 font_cnt 个字体
        selected_font_ids = random.sample(range(len(image_maps)), font_cnt)
        selected_image_maps = [image_maps[font_id] for font_id in selected_font_ids]

        for font_map in selected_image_maps:
            for _ in range(sample_cnt):
                while True:
                    start_idx = random.randint(0, text_length - 25)
                    selected_chars = self.text[start_idx:start_idx + 25]
                    if all(char in self.chars_set for char in selected_chars):
                        break

                image = Image.new("L", (224, 224), color=255)
                char_size = 42
                spacing = 2

                for i, char in enumerate(selected_chars):
                    x = spacing + (i % 5) * (char_size + spacing)
                    y = spacing + (i // 5) * (char_size + spacing)
                    char_image = font_map.get(char)
                    if char_image:
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
                        char_image = char_image.resize((char_size, char_size), Image.LANCZOS)
                        image.paste(char_image, (x, y))
                samples.append(image)
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
    fonts_dir = "./font_ds/fonts"  # 字体文件夹路径
    text_file = "./font_ds/text.txt"  # 文本文件路径
    chars_file = "./font_ds/chars.txt"  # 常用字文件路径
    output_dir = "sample"  # 输出文件夹路径

    sampler = FontSampler(fonts_dir, text_file, chars_file, max_fonts=16, font_size=76, train_ratio=0.5)
    train_samples = sampler.sample(font_cnt=4, sample_cnt=8, sample_source="train")
    test_samples = sampler.sample(font_cnt=4, sample_cnt=8, sample_source="test")
    sampler.save_samples(train_samples, os.path.join(output_dir, "train"))
    sampler.save_samples(test_samples, os.path.join(output_dir, "test"))