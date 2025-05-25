import os
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion

# 定义输入和输出文件夹路径
raw_data_dir = 'candidates'
scaled_data_dir = 'scaled_candidates'

# 确保输出文件夹存在
os.makedirs(scaled_data_dir, exist_ok=True)

def compute_stroke_width(np_gray):
    """
    将灰度图二值化：亮度 > 128 置为 255（背景），其余置为 0（笔画）。
    利用距离变换计算笔画区域的局部距离，笔画宽度约为 2 * 平均距离。
    """
    # 二值化
    binary = np.array(np_gray.point(lambda p: 255 if p > 128 else 0))
    # 构造笔画 mask：笔画部分为 True
    stroke_mask = (binary == 0)
    if np.sum(stroke_mask) == 0:
        return 0
    # distance_transform_edt 计算每个“零”元素（即笔画区域在背景为非零时）的距离
    # 为此，我们先反转 mask：笔画部分变为 0，背景为 1
    distances = distance_transform_edt(stroke_mask)
    # 平均笔画宽度近似为 2 倍边缘到背景的平均距离
    avg_width = 2 * np.mean(distances[stroke_mask])
    return avg_width

# 第一遍遍历所有图片，计算各自笔画宽度
stroke_widths = []
filenames = []
for filename in os.listdir(raw_data_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(raw_data_dir, filename)
        try:
            with Image.open(input_path) as img:
                img_gray = img.convert('L')
                width, height = img_gray.size
                # 不做裁剪，只计算整幅图的笔画宽度
                avg_width = compute_stroke_width(img_gray)
                print(avg_width, filename)
                if avg_width > 0:
                    stroke_widths.append(avg_width)
                    filenames.append(filename)
        except Exception as e:
            print(f"处理图片 {filename} 失败: {e}")

if len(stroke_widths) == 0:
    print("没有计算出有效的笔画宽度。")
    exit(1)

global_avg = np.mean(stroke_widths)
print(f"全局平均笔画宽度为: {global_avg:.2f}")

def iterative_erode(norm_img, global_avg, eps, alpha=2):
    """
    侵蚀操作：输入 norm_img 为 [0,1] 范围内的灰度图，
    每次先进行 3x3 高斯模糊，再对图像每个元素做 alpha 次方（alpha=2），
    循环直到计算出的笔画宽度小于 global_avg+eps。
    """
    while True:
        # 将归一化结果转换为 0-255 的8位图，传入 compute_stroke_width 计算笔画宽度
        pil_temp = Image.fromarray((norm_img * 255).astype(np.uint8), mode='L')
        current_width = compute_stroke_width(pil_temp)
        # print(f"    erode 当前笔画宽度: {current_width:.2f}, {global_avg + eps}")
        if current_width < global_avg + eps:
            break
        blurred = cv2.GaussianBlur(norm_img, (3, 3), 0.25)
        norm_img = np.power(blurred, 1.0/alpha)

    # print(f"erode 处理后的笔画宽度: {current_width:.2f}")

    return norm_img


def iterative_dilate(norm_img, global_avg, eps, alpha=2):
    """
    膨胀操作：输入 norm_img 为 [0,1] 的灰度图，
    每次先进行 3x3 高斯模糊，再对每个像素做 alpha 次方（alpha=0.5），
    循环直到计算出的笔画宽度大于 global_avg-eps。
    """
    while True:
        pil_temp = Image.fromarray((norm_img * 255).astype(np.uint8), mode='L')
        current_width = compute_stroke_width(pil_temp)
        # print(f"    dilate 当前笔画宽度: {current_width:.2f}")
        if current_width > global_avg - eps:
            break
        blurred = cv2.GaussianBlur(norm_img, (3, 3), 0.25)
        norm_img = np.power(blurred, alpha)

    # print(f"dilate 处理后的笔画宽度: {current_width:.2f}")

    return norm_img

# 第二遍处理每张图片，对笔画宽度进行调整，然后中心裁剪及缩放
for filename in os.listdir(raw_data_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        input_path = os.path.join(raw_data_dir, filename)
        try:
            with Image.open(input_path) as img:
                # 转为灰度
                img_gray = img.convert('L')
                # 二值化：背景 255，笔画 0
                binary = img_gray.point(lambda p: 255 if p > 128 else 0)

                # 裁剪图片使其变为正方形（使用调整后的图）
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
                binary = binary.crop((left, upper, right, lower))

                np_binary = np.array(binary, dtype=np.uint8
                # # 计算当前图片笔画宽度
                # current_width = compute_stroke_width(img_gray)
                # print(f"当前笔画宽度: {current_width:.2f}, {filename}")
                
                # # 归一化到 [0,1]；注意 np_binary 原本只有0和255
                # norm_img = np_binary.astype(np.float32) / 255.0
                
                # eps = 0.1
                # # 如果当前笔画宽度过大，则执行侵蚀，使其减小
                # if current_width > global_avg + eps:
                #     norm_img = iterative_erode(norm_img, global_avg, eps, alpha=1.5)
                # # 如果当前笔画宽度过小，则执行膨胀，使其增大
                # elif current_width < global_avg - eps:
                #     norm_img = iterative_dilate(norm_img, global_avg, eps, alpha=1.5)
                
                # # 转换回 0-255 的8位图像
                # np_binary = (norm_img * 255).astype(np.uint8)
                # img_adjusted = Image.fromarray(np_binary, mode='L')

                # current_width = compute_stroke_width(img_adjusted)
                # print(f"处理后笔画宽度: {current_width:.2f}, {filename}")

                # 缩小图片到 224x224
                img_resized = np_binary.resize((224, 224), Image.LANCZOS)

                output_path = os.path.join(scaled_data_dir, os.path.splitext(filename)[0] + '.png')
                img_resized.save(output_path)
        except Exception as e:
            print(f"处理图片 {filename} 失败: {e}")

print(f"所有图片已处理并保存到 {scaled_data_dir} 文件夹中。")