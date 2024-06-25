import os
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
import json

def process_file(args):
    file_path, map_dir, threshold = args
    # 读取激活图
    activation_map = np.load(file_path)

    # 计算超过阈值的百分比
    total_elements = activation_map.size
    count_above_threshold = np.sum(activation_map > threshold)
    percentage = (count_above_threshold / total_elements) * 100

    # 获取类别和层次信息
    parts = os.path.relpath(file_path, map_dir).split(os.sep)
    if len(parts) >= 3:
        category = parts[0]
        layer = parts[2].split('_')[0]  # 提取layer1, layer2等层次信息
        return (category, layer, percentage)
    return None

def calculate_percentage_above_threshold(map_dir, threshold, num_processes=20):
    """
    计算给定目录中每个类别的每个层次激活图中超过阈值的百分比
    :param map_dir: 存储激活图的目录
    :param threshold: 激活值阈值
    :param num_processes: 使用的进程数量
    :return: 包含每个类别每个层次激活图超过阈值百分比的字典
    """
    percentages = defaultdict(lambda: defaultdict(list))
    tasks = []

    # 遍历目录中的所有.npy文件
    for root, _, files in os.walk(map_dir):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                tasks.append((file_path, map_dir, threshold))

    # 并行处理文件
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_file, tasks), total=len(tasks), desc="Processing activation maps"))

    # 聚合结果
    for result in results:
        if result:
            category, layer, percentage = result
            percentages[category][layer].append(percentage)

    # 计算每个类别每个层次的平均百分比
    for category in percentages:
        for layer in percentages[category]:
            percentages[category][layer] = np.mean(percentages[category][layer])

    return percentages

def save_percentages(percentages, file_path):
    with open(file_path, 'w') as f:
        json.dump(percentages, f)

def load_percentages(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def plot_activation_percentages_by_layer(percentages, output_dir):
    """
    生成每个层的激活值大于阈值的柱状图，每个图展示所有类别
    :param percentages: 包含每个类别每个层次激活图超过阈值百分比的字典
    :param output_dir: 保存柱状图的目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categories = sorted(percentages.keys())
    layers = sorted(next(iter(percentages.values())).keys())  # 假设所有类别都有相同的层次

    for layer in layers:
        x = np.arange(len(categories))
        y = [percentages[category][layer] for category in categories]
        width = 0.35  # 每个柱状图的宽度

        fig, ax = plt.subplots(figsize=(40, 10))  # 调整图像的长宽比
        ax.bar(x, y, width, label=layer)

        ax.set_xlabel('Categories')
        ax.set_ylabel('Percentage of Activation > Threshold')
        ax.set_title(f'Percentage of Activations Exceeding Threshold for {layer}')
        ax.set_xticks(x)
        # 不显示横轴标签
        ax.set_xticklabels([])
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{layer}_activation_percentages.pdf"))
        plt.close()

# 示例使用
activation_map_directory = "/home/nus-wk/activation_map_of_IN1K/maps_of_in1k"
threshold_value = 0.5
percentages_file = "percentages.json"

# 检查是否已存在百分比文件，如果存在则加载，否则计算并保存
if os.path.exists(percentages_file):
    percentages = load_percentages(percentages_file)
else:
    percentages = calculate_percentage_above_threshold(activation_map_directory, threshold_value, num_processes=40)
    save_percentages(percentages, percentages_file)

# 生成每个层的柱状图
output_bar_chart_dir = "activation_percentages_by_layer"
plot_activation_percentages_by_layer(percentages, output_bar_chart_dir)

# 打印结果
for category, layers in percentages.items():
    print(f"Category: {category}")
    for layer, percentage in layers.items():
        print(f"  {layer}: {percentage:.2f}%")
