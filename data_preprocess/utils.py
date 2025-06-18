import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from PIL import Image
import os
import requests
import matplotlib.pyplot as plt


# import torch
# from torchvision import transforms
# from PIL import Image
# import os
#
# # 定义预处理操作
# preprocess = transforms.Compose([
#     transforms.Resize(256),  # 调整图像大小
#     transforms.CenterCrop(224),  # 裁剪到中心区域
#     transforms.ToTensor(),  # 转换为张量
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
# ])
#
# def load_and_preprocess_image(image_path):
#     image = Image.open(image_path).convert("RGB")  # 打开图像并转换为RGB格式
#     image = preprocess(image)  # 进行预处理
#     return image
#
# # 示例：加载和预处理一个图像
# image_path = 'F:/data/VISPR/images/val2017/2017_10356759.jpg'
# image_tensor = load_and_preprocess_image(image_path)
#
# print(image_tensor.shape)  # 打印图像张量的形状以验证
# print(image_tensor)

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")

import os
import csv


# def clean_csv_folder(folder_path):
#     """
#     检测文件夹内的所有CSV文件，如果第一行是`nude_class,score,box`，就删除该文件。
#     同时统计文件夹内所有表格的数量以及删除的表格数量。
#     """
#     if not os.path.exists(folder_path):
#         raise FileNotFoundError(f"路径 {folder_path} 不存在.")
#
#     total_csv_files = 0
#     deleted_csv_files = 0
#
#     # 遍历文件夹内的所有文件
#     for file in os.listdir(folder_path):
#         # 检查是否是CSV文件
#         if file.lower().endswith('.csv'):
#             total_csv_files += 1
#             file_path = os.path.join(folder_path, file)
#
#             try:
#                 # 打开CSV文件，读取第一行
#                 with open(file_path, 'r') as csvfile:
#                     reader = csv.reader(csvfile)
#                     first_row = next(reader, None)
#
#                     # 如果第一行是指定的列名，删除该文件
#                     if first_row == ['nude_class', 'score', 'box']:
#                         os.remove(file_path)
#                         deleted_csv_files += 1
#                         print(f"已删除表格: {file_path}")
#             except Exception as e:
#                 print(f"处理文件 {file_path} 时发生错误: {e}")
#
#     # 打印统计结果
#     print(f"总表格数: {total_csv_files}")
#     print(f"删除表格数: {deleted_csv_files}")
#
#
# # 使用函数
# folder_path = "/home/liangxy/pycharm/sg_privacy/data/privacy_alert_v2/SC/train"  # 替换为目标文件夹路径
# clean_csv_folder(folder_path)


import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix

# 输入混淆矩阵
conf_matrix = np.array([
    [291, 33],  # 第一类的真阳性、假阴性
    [50, 378]   # 第二类的假阳性、真阴性
])

# 计算每个类别的指标
precision_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[1, 0])
recall_0 = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)

precision_1 = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
recall_1 = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)

# 加权平均值（可以根据类别权重计算）
support_0 = np.sum(conf_matrix[0])  # 第一类的样本数
support_1 = np.sum(conf_matrix[1])  # 第二类的样本数
total_support = support_0 + support_1

weighted_precision = (precision_0 * support_0 + precision_1 * support_1) / total_support
weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total_support
weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total_support

# 计算未加权平均准确度
macro_accuracy = (precision_0 + precision_1) / 2
macro_f1 = (f1_0 + f1_1) / 2

# 打印结果
print("分类 0:")
print(f"  Precision: {precision_0:.4f}")
print(f"  Recall: {recall_0:.4f}")
print(f"  F1 Score: {f1_0:.4f}")

print("分类 1:")
print(f"  Precision: {precision_1:.4f}")
print(f"  Recall: {recall_1:.4f}")
print(f"  F1 Score: {f1_1:.4f}")

print("加权平均:")
print(f"  Weighted Precision: {weighted_precision:.4f}")
print(f"  Weighted Recall: {weighted_recall:.4f}")
print(f"  Weighted F1 Score: {weighted_f1:.4f}")

print("未加权平均:")
print(f"  Macro Accuracy: {macro_accuracy:.4f}")
print(f"  Macro F1 Score: {macro_f1:.4f}")