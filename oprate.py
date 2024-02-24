import cv2
import os
from werkzeug.utils import secure_filename


# 图片前置处理

# 图片反色函数
def invert_images(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹中的所有图像文件
    image_files = os.listdir(input_folder)

    for image_file in image_files:
        # 构建输入和输出文件的路径
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # 读取图像
        image = cv2.imread(input_path)

        # 反转图像
        inverted_image = cv2.bitwise_not(image)

        # 保存反转后的图像
        cv2.imwrite(output_path, inverted_image)


# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////


# 50x倍率图片处理
# 生成前景图像
def generate_foreground_50x(path, kernel_size):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 定义大内核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 进行开运算
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    return opened_image


# 直方图阈值分割
def apply_histogram_iteration_50x(path):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

    # 初始化阈值
    threshold_value = 128
    prev_threshold = 0

    while abs(threshold_value - prev_threshold) > 1:
        prev_threshold = threshold_value

        # 根据当前阈值分割图像
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 计算两个区域的平均灰度值
        foreground_mean = cv2.mean(gray_image, mask=thresholded_image)[0]
        background_mean = cv2.mean(gray_image, mask=cv2.bitwise_not(thresholded_image))[0]

        # 更新阈值为两个区域的平均灰度值的均值
        threshold_value = int((foreground_mean + background_mean) / 2)
    threshold_value = threshold_value  # +20
    # 应用最终阈值分割
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return thresholded_image


# 闭运算
def close_image_50x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 对图像进行膨胀操作
    dilated_image = cv2.dilate(binary_image, kernel)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(dilated_image, kernel)

    return eroded_image


# 开运算
def remove_isolated_pixels_50x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(binary_image, kernel)

    # 对腐蚀后的图像进行膨胀操作
    dilated_image = cv2.dilate(eroded_image, kernel)

    # 通过与原始二值图像进行按位与操作，找到孤立点的位置
    isolated_pixels = cv2.bitwise_and(binary_image, cv2.bitwise_not(dilated_image))

    # 将孤立点设置为黑色
    cleaned_image = cv2.bitwise_xor(binary_image, isolated_pixels)

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(cleaned_image, (5, 5), 0)

    return cleaned_image


def image_50x(file):
    # 将上传的文件保存到临时目录中
    filename = secure_filename(file.filename)
    temp_filepath = f'temp/{filename}'
    file.save(temp_filepath)

    # 读取图像文件
    img = cv2.imread(temp_filepath)

    # 定义大内核大小
    kernel_size_F = (20, 20)
    # 生成前景图像
    foreground_image_50x = generate_foreground_50x(temp_filepath, kernel_size_F)
    # 获取处理后的图像
    thresholded_image = apply_histogram_iteration_50x(temp_filepath)
    # 相减操作
    subtracted_image = cv2.subtract(thresholded_image, foreground_image_50x, dst=None, mask=None, dtype=None)
    # 对图像进行闭运算
    kernel_size_C = (5, 5)
    image_C = close_image_50x(subtracted_image, kernel_size_C)
    # 开运算以消除孤立点和断开连接
    # 定义结构元素大小
    kernel_size_O_0 = (4, 4)
    kernel_size_O_1 = (6, 6)
    # 消除孤立点
    cleaned_image = remove_isolated_pixels_50x(image_C, kernel_size_O_0)
    image_50x = remove_isolated_pixels_50x(cleaned_image, kernel_size_O_1)
    # cleaned_image = cv2.medianBlur_50x(cleaned_image, 7)
    return image_50x


# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////

# 20x图片处理


def generate_foreground_20x(path, kernel_size):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 定义大内核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 进行开运算
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    return opened_image


# 直方图阈值分割
def apply_histogram_iteration_20x(path):
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    # 相减操作
    #     subtracted_image = cv2.subtract(gray_image, foreground_image, dst=None, mask=None, dtype=None)
    # 反转图像
    #     inverted_image = cv2.bitwise_not(subtracted_image)

    # 初始化阈值
    threshold_value = 128
    prev_threshold = 0

    while abs(threshold_value - prev_threshold) > 1:
        prev_threshold = threshold_value

        # 根据当前阈值分割图像
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 计算两个区域的平均灰度值
        foreground_mean = cv2.mean(gray_image, mask=thresholded_image)[0]
        background_mean = cv2.mean(gray_image, mask=cv2.bitwise_not(thresholded_image))[0]

        # 更新阈值为两个区域的平均灰度值的均值
        threshold_value = int((foreground_mean + background_mean) / 2)
    threshold_value = threshold_value  # + 20
    # 应用最终阈值分割
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    return thresholded_image


# 图像闭操作连接部分断开的边缘
def close_image_20x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 对图像进行膨胀操作
    dilated_image = cv2.dilate(binary_image, kernel)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(dilated_image, kernel)

    return eroded_image


# 开运算以消除孤立点
def remove_isolated_pixels_20x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(binary_image, kernel)

    # 对腐蚀后的图像进行膨胀操作
    dilated_image = cv2.dilate(eroded_image, kernel)

    # 通过与原始二值图像进行按位与操作，找到孤立点的位置
    isolated_pixels = cv2.bitwise_and(binary_image, cv2.bitwise_not(dilated_image))

    # 将孤立点设置为黑色
    cleaned_image = cv2.bitwise_xor(binary_image, isolated_pixels)

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(cleaned_image, (5, 5), 0)

    return cleaned_image


def image_20x(file):
    # 将上传的文件保存到临时目录中
    filename = secure_filename(file.filename)
    temp_filepath = f'temp/{filename}'
    file.save(temp_filepath)

    # 读取图像文件
    img = cv2.imread(temp_filepath)
    # 定义大内核大小
    kernel_size_F = (35, 35)

    # 生成前景图像
    foreground_image = generate_foreground_20x(img, kernel_size_F)
    # 获取处理后的图像
    thresholded_image = apply_histogram_iteration_20x(img)
    # 相减操作
    subtracted_image = cv2.subtract(thresholded_image, foreground_image, dst=None, mask=None, dtype=None)

    kernel_size_C = (8, 8)
    image_C = close_image_20x(subtracted_image, kernel_size_C)

    # 定义结构元素大小
    kernel_size_O = (4, 4)
    kernel_size_O1 = (2, 2)
    # kernel_size_O2 = (6, 6)
    # 消除孤立点
    cleaned_image = remove_isolated_pixels_20x(image_C, kernel_size_O)
    image_20x = remove_isolated_pixels_20x(cleaned_image, kernel_size_O1)
    return image_20x


# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////

# 10x图片处理

def generate_foreground_10x(path, kernel_size):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)

    # 二值化处理
    _, thresholded_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 定义大内核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 进行开运算
    opened_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)

    return opened_image


# 直方图阈值分割
def apply_histogram_iteration_10x(path):
    gray_image = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)
    # 相减操作
    #     subtracted_image = cv2.subtract(gray_image, foreground_image, dst=None, mask=None, dtype=None)
    # 反转图像
    #     inverted_image = cv2.bitwise_not(subtracted_image)

    # 初始化阈值
    threshold_value = 128
    prev_threshold = 0

    while abs(threshold_value - prev_threshold) > 1:
        prev_threshold = threshold_value

        # 根据当前阈值分割图像
        _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 计算两个区域的平均灰度值
        foreground_mean = cv2.mean(gray_image, mask=thresholded_image)[0]
        background_mean = cv2.mean(gray_image, mask=cv2.bitwise_not(thresholded_image))[0]

        # 更新阈值为两个区域的平均灰度值的均值
        threshold_value = int((foreground_mean + background_mean) / 2)
    threshold_value = threshold_value  # + 15
    # 应用最终阈值分割
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded_image


# 图像闭操作连接部分断开的边缘
def close_image_10x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # 对图像进行膨胀操作
    dilated_image = cv2.dilate(binary_image, kernel)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(dilated_image, kernel)

    return eroded_image


# 开运算以消除孤立点
def remove_isolated_pixels_10x(binary_image, kernel_size):
    # 定义结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # 对图像进行腐蚀操作
    eroded_image = cv2.erode(binary_image, kernel)

    # 对腐蚀后的图像进行膨胀操作
    dilated_image = cv2.dilate(eroded_image, kernel)

    # 通过与原始二值图像进行按位与操作，找到孤立点的位置
    isolated_pixels = cv2.bitwise_and(binary_image, cv2.bitwise_not(dilated_image))

    # 将孤立点设置为黑色
    cleaned_image = cv2.bitwise_xor(binary_image, isolated_pixels)

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(cleaned_image, (5, 5), 0)

    return cleaned_image


def image_10x(file):
    # 将上传的文件保存到临时目录中
    filename = secure_filename(file.filename)
    temp_filepath = f'temp/{filename}'
    file.save(temp_filepath)

    # 读取图像文件
    img = cv2.imread(temp_filepath)
    # 定义大内核大小
    kernel_size_F = (35, 35)

    # 生成前景图像
    foreground_image = generate_foreground_10x(img, kernel_size_F)
    # 获取处理后的图像
    thresholded_image = apply_histogram_iteration_10x(img)
    # 相减操作
    subtracted_image = cv2.subtract(thresholded_image, foreground_image, dst=None, mask=None, dtype=None)

    kernel_size_C = (6, 6)
    image_C = close_image_10x(subtracted_image, kernel_size_C)
    # 定义结构元素大小
    kernel_size_O = (5, 5)
    kernel_size_O1 = (3, 3)
    # 消除孤立点
    cleaned_image = remove_isolated_pixels_10x(image_C, kernel_size_O)
    image_10x = remove_isolated_pixels_10x(cleaned_image, kernel_size_O1)
    return image_10x


# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////////


# 调用cellpose接口实现细胞计数
from cellpose import utils, io, models
from cellpose import plot
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO


# %matplotlib inline

# def cell_count(img):
#     # 执行细胞计数操作（你的 cell_count 函数的部分）
#     mpl.rcParams['figure.dpi'] = 300
#     model = models.Cellpose(gpu=True, model_type='cyto')
#     chan = [2, 3]
#     masks, flows, styles, diams = model.eval(img, diameter=None, channels=chan)
#
#     # 创建一个 Matplotlib 图像
#     fig = plt.figure(figsize=(12, 5))
#     plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
#     plt.tight_layout()
#
#     # 将图像转换为字节流
#     image_data = BytesIO()
#     plt.savefig(image_data, format='png')
#     plt.close(fig)
#     image_data.seek(0)
#
#     # 计算细胞数
#     cell_count = str(masks.max())
#
#     # 返回图像数据和细胞数作为响应
#     return Response(image_data, content_type='image/png'), cell_count


def cell_count_10x(img):
    # 执行细胞计数操作
    mpl.rcParams['figure.dpi'] = 300
    model = models.Cellpose(gpu=True, model_type='cyto')
    chan = [2, 3]
    # diameter预设细胞直径
    masks, flows, styles, diams = model.eval(img, diameter=20, channels=chan)

    # 创建一个 Matplotlib 图像
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
    plt.tight_layout()

    # 将图像转换为字节流
    image_data = BytesIO()
    plt.savefig(image_data, format='png')
    plt.close(fig)
    image_data.seek(0)

    # 计算细胞数
    cell_count = str(masks.max())

    # 返回图像数据和细胞数
    return image_data, cell_count


def cell_count_20x(img):
    # 执行细胞计数操作
    mpl.rcParams['figure.dpi'] = 300
    model = models.Cellpose(gpu=True, model_type='cyto')
    chan = [2, 3]
    masks, flows, styles, diams = model.eval(img, diameter=40, channels=chan)

    # 创建一个 Matplotlib 图像
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
    plt.tight_layout()

    # 将图像转换为字节流
    image_data = BytesIO()
    plt.savefig(image_data, format='png')
    plt.close(fig)
    image_data.seek(0)

    # 计算细胞数
    cell_count = str(masks.max())

    # 返回图像数据和细胞数
    return image_data, cell_count


def cell_count_50x(img):
    # 执行细胞计数操作
    mpl.rcParams['figure.dpi'] = 300
    model = models.Cellpose(gpu=True, model_type='cyto')
    chan = [2, 3]
    masks, flows, styles, diams = model.eval(img, diameter=100, channels=chan)

    # 创建一个 Matplotlib 图像
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=chan)
    plt.tight_layout()

    # 将图像转换为字节流
    image_data = BytesIO()
    plt.savefig(image_data, format='png')
    plt.close(fig)
    image_data.seek(0)

    # 计算细胞数
    cell_count = str(masks.max())

    # 返回图像数据和细胞数
    return image_data, cell_count
