import os
import random
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import cv2

class FontSampler:
    def __init__(self, fonts_dir, text_file, chars_file, font_size=78):
        """
        初始化 FontSampler 类。

        :param fonts_dir: 字体文件夹路径，包含 train 和 test 子文件夹
        :param text_file: 文本文件路径
        :param chars_file: 常用字文件路径
        :param font_size: 字体大小
        """
        # 从 train 和 test 文件夹中加载字体
        self.train_font_files = self.load_fonts(os.path.join(fonts_dir, "train"))
        self.test_font_files = self.load_fonts(os.path.join(fonts_dir, "test"))

        # 加载文本和字符集
        self.text = self.load_text(text_file)
        self.chars = self.load_text(chars_file).strip()
        self.chars_set = set(self.chars)

        # 为 train 和 test 字体生成图像 map
        self.train_image_maps = self.generate_image_map(self.train_font_files, self.chars, font_size)
        self.test_image_maps = self.generate_image_map(self.test_font_files, self.chars, font_size)

    @staticmethod
    def load_fonts(fonts_dir):
        """
        加载指定文件夹中的所有 ttf 字体文件。

        :param fonts_dir: 字体文件夹路径
        :return: 字体路径列表
        """
        font_files = [os.path.join(fonts_dir, f) for f in os.listdir(fonts_dir) if f.endswith('.ttf')]
        random.shuffle(font_files)  # 随机打乱字体顺序
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

    def generate_image_map(self, font_files, chars, font_size):
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

    def sample(self, font_cnt, sample_cnt, 
               rotation_range=(-4, 4), translation_range=(-2, 2), 
               font_scale_range=(0.85, 1.06), char_scale_range=(0.95,1.0), 
               bold_range=(-1, 1), sample_source="train"):
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

        # 根据 sample_source 选择对应的字体图像 map
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
                font_scale = random.uniform(font_scale_range[0], font_scale_range[1])
                bold_effect = round(random.uniform(bold_range[0], bold_range[1]))

                for i, char in enumerate(selected_chars):
                    x = spacing + (i % 5) * (char_size + spacing)
                    y = spacing + (i // 5) * (char_size + spacing)
                    char_image = font_map.get(char)
                    if char_image:
                        
                        # 转换为 NumPy 数组
                        char_np = np.array(char_image)
                        
                        # 对字符进行随机加粗或侵蚀
                        if bold_effect > 0:
                            kernel = np.ones((bold_effect+1, bold_effect+1), dtype=np.uint8)
                            char_np = cv2.dilate(char_np, kernel, iterations=1)
                        elif bold_effect < 0:
                            kernel = np.ones((-bold_effect+1, -bold_effect+1), dtype=np.uint8)
                            char_np = cv2.erode(char_np, kernel, iterations=1)
                        
                        # 转回 PIL Image
                        char_image = Image.fromarray(char_np)
    
                        angle = random.uniform(rotation_range[0], rotation_range[1])
                        char_image = char_image.rotate(angle, expand=True, fillcolor=255)
                        
                        # 缩放字体图像，但保持画布大小为 84x84
                        char_scale = font_scale * random.uniform(char_scale_range[0], char_scale_range[1])
                        char_scale = min(char_scale, 1.0)  # 限制缩放范围
                        new_size = (int(84 * char_scale), int(84 * char_scale))
                        char_image = char_image.resize(new_size, Image.LANCZOS)

                        # 创建固定大小的画布
                        canvas_fixed = Image.new("L", (84, 84), color=255)
                        paste_x = (84 - new_size[0]) // 2
                        paste_y = (84 - new_size[1]) // 2
                        canvas_fixed.paste(char_image, (paste_x, paste_y))
                        char_image = canvas_fixed

                        dx = random.uniform(translation_range[0], translation_range[1])
                        dy = random.uniform(translation_range[0], translation_range[1])
                        x += round(dx)
                        y += round(dy)

                        char_image = char_image.resize((char_size, char_size), Image.LANCZOS)
                        image.paste(char_image, (x, y))
                samples.append(image)
        return samples

    @staticmethod
    def save_samples(samples, output_dir, sample_cnt):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for idx, image in enumerate(samples):
            # 每 sample_cnt 张属于同一字体
            font_label = idx // sample_cnt
            inner_idx = idx % sample_cnt
            output_path = os.path.join(output_dir, f"font_{font_label}_sample_{inner_idx}.png")
            image.save(output_path)

# 测试代码
if __name__ == "__main__":
    fonts_dir = "./font_ds_mini/fonts"  # 字体文件夹路径
    text_file = "./font_ds/cleaned_text.txt"  # 文本文件路径
    chars_file = "./font_ds/chars.txt"  # 常用字文件路径
    output_dir = "sample"  # 输出文件夹路径

    sampler = FontSampler(fonts_dir, text_file, chars_file, font_size=76)
    test_samples = sampler.sample(font_cnt=16, sample_cnt=16, sample_source="test")
    sampler.save_samples(test_samples, os.path.join(output_dir, "test"), sample_cnt=16)