import numpy as np
import torch
import os
from main import parse_arguments
from model.SG_Transformer import TransformerSG
from sg_utils.sg_util import load_data_from_npy, calculate_iou, get_font_size
from data_preprocess.dataset import SceneGraphDataset, DataLoader, collate_fn
from ExplanationGenerator import Generator
from captum.attr import visualization
import json
import cv2
import matplotlib.pyplot as plt
import math
import time

args = parse_arguments()

"""
输出图片的解释。
"""

def calculate_contributions(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for features, adjacency_matrix, labels, subgraph_indices_list, padded_extra_matrices in data_loader:
            features, adjacency_matrix, labels, extra_matrix = features.to(device), adjacency_matrix.to(device), labels.to(device), padded_extra_matrices.to(device)
            features.requires_grad = True
            # 前向传播
            output, attn_weights = model(features, adjacency_matrix, subgraph_indices_list, padded_extra_matrices)

            output = output.squeeze(0)
            label = output.argmax().item()

            one_hot_output = torch.zeros(output.size()).to(output.device)
            one_hot_output[label] = 1
            model.zero_grad()
            output.backward(gradient=one_hot_output)

            contributions = features.grad
            contributions = contributions.mean(dim=0)
    return contributions, label


def calculate_attention_contributions(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for features, adjacency_matrix, labels, subgraph_indices_list, padded_extra_matrices in data_loader:
            features, adjacency_matrix, labels, extra_matrix = features.to(device), adjacency_matrix.to(device), labels.to(device), padded_extra_matrices.to(device)

            # 前向传播
            outputs, attn_weights = model(features, adjacency_matrix, subgraph_indices_list, padded_extra_matrices)

            # 从注意力权重中提取信息
            batch_size = attn_weights[0].size(0)  # 取第一个层的大小
            subgraph_attn_weights = []
            for i in range(batch_size):
                batch_attn_weights = []
                for subgraph in subgraph_indices_list[i]:
                    subgraph_attn = torch.stack(
                        [attn_weights[layer][i, subgraph, :].mean(dim=0) for layer in range(len(attn_weights))])
                    batch_attn_weights.append(subgraph_attn.mean(dim=0))
                subgraph_attn_weights.append(torch.stack(batch_attn_weights))
    return subgraph_attn_weights


def create_relation_dict(item):
    base_filename = item['filename']
    node_names = []
    node_positions = []
    # 遍历 r_pair 列表，创建三元组
    for relation in item['r_pair']:
        # 获取三元组中节点和关系的名称
        node1_name = item['n_list'][relation[0]]
        node2_name = item['n_list'][relation[1]]
        relation_name = item['r_list'][relation[2]]

        # 获取节点的位置
        node1_position = item['n_position'][relation[0]]
        node2_position = item['n_position'][relation[1]]

        node_name = [node1_name, relation_name, node2_name]
        node_position = [node1_position, node2_position]
        node_names.append(node_name)
        node_positions.append(node_position)

    # 创建字典
    relation_dict = {
        "filename": base_filename,
        "pair_name": node_names,
        "pair_position": node_positions
    }

    return relation_dict


# 定义函数来提取数据并进行贡献分析
def analyze_contributions_by_filename(model, filename, data_path, json_path, device):
    # 加载数据
    print("load file:"+filename)
    data = load_data_from_npy(data_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    # 遍历 JSON 数据列表，查找匹配的 filename
    for item in json_data:
        if item['filename'] == filename:
            data_dict = create_relation_dict(item)
            break

    one_data = []
    # 根据 filename 查找对应的数据条目
    entry = next(item for item in data if item['filename'] == filename)
    one_data.append(entry)
    process_data = np.array(one_data)

    test_dataset = SceneGraphDataset(process_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    label = entry["label"]
    lrp_explanation(model, device, test_loader, label, data_dict)


    # # 计算梯度贡献
    # contributions, predicted_label = calculate_contributions(model, test_loader, device)
    # print(f"Predicted label: {predicted_label}, True label: {label}")
    # print(f"Subgraph contributions: {contributions}")
    #
    # # 计算注意力贡献
    # attn_contributions = calculate_attention_contributions(model, test_loader, device)
    # print(f"Attention-based contributions: {attn_contributions}")

def lrp_explanation(model, device, data_loader, label, data_dict):
    # initialize the explanations generator
    explanations = Generator(model)

    classifications = ["public", "privacy"]
    r_pair = data_dict['pair_name']

    true_class = label
    test_loader = data_loader
    # generate an explanation for the input
    expl, predict_label = explanations.generate_LRP(data_loader, device, start_layer=0)
    # expl, predict_label = generate_LRP(model, data_loader, device, start_layer=0)[0]
    # normalize scores
    expl = (expl - expl.min()) / (expl.max() - expl.min())
    # get class name
    class_name = classifications[predict_label]
    true_class_name = classifications[true_class]
    # if the classification is negative, higher explanation scores are more negative
    # flip for visualization
    # if class_name == "public":
    #     expl *= (-1)
    # tokens = tokenizer.convert_ids_to_tokens(input_ids.flatten())
    print("predict label: ", class_name)
    print("true label: ", true_class_name)
    print([(r_pair[i], expl[0,i+1].item()) for i in range(len(r_pair))])
    visualize_scores(data_dict, expl)


def visualize_scores(data_dict, expl):
    # 读取图像
    filename = data_dict['filename']
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    image_path = os.path.join(dataset_dir, f"images/{args.data_type}/{filename}.jpg")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取pair_name和对应的分数，并按分数排序
    r_pair = data_dict['pair_name']
    scores_with_pairs = [(r_pair[i], expl[0, i + 1].item()) for i in range(len(r_pair))]
    sorted_scores_with_pairs = sorted(scores_with_pairs, key=lambda x: x[1], reverse=True)

    # 获取位置信息
    positions = data_dict['pair_position']
    # 创建一个透明的图层
    overlay = np.zeros_like(image, dtype=np.float32)

    # 设置IoU的阈值，如果两个区域的IoU大于此阈值，则认为它们重合
    iou_threshold = 0.5

    # 使用字典来记录每个位置的最大分数
    max_scores = {}

    for i, pos_pair in enumerate(positions):
        score = expl[0, i + 1].item()
        if math.isnan(score):
            score = 1
        pair_name =r_pair[i]
        # # 改变相应的隐私分数
        # for j in pair_name:
        #     if j == "man" or j == "woman":
        #         score = 1
        #     elif j == "license plate":
        #         score = 0
        for pos in pos_pair:
            x1, y1, x2, y2 = map(int, map(float, pos))
            box = (x1, y1, x2, y2)
            # 检查该框是否与已有框重叠
            found_overlap = False
            for existing_box in list(max_scores.keys()):
                if calculate_iou(box, existing_box) > iou_threshold:
                    # 如果重叠，取分数最高的
                    max_scores[existing_box] = max(max_scores[existing_box], score)
                    found_overlap = True
                    break

            if not found_overlap:
                # 如果没有重叠，添加新框
                max_scores[box] = score

    # 根据最高分数更新透明层
    for box, max_score in max_scores.items():
        x1, y1, x2, y2 = box
        intensity = (max_score ** 2) * 255
        overlay[y1:y2, x1:x2] = max_score * intensity * np.array([1, 1, 1], dtype=np.float32)

        # 将叠加图像与原始图像合并
    alpha = 0.5  # 透明度系数
    marked_image = cv2.addWeighted(overlay, alpha, image.astype(np.float32), 1 - alpha, 0)

    # 绘制结果图像
    fig, ax = plt.subplots(figsize=(16, 8))

    # 显示标记图像
    ax.imshow(marked_image.astype(np.uint8))
    ax.axis('off')


    # 在右侧显示分数文本
    for i, (pair, score) in enumerate(sorted_scores_with_pairs):
        text_y = (i + 0.5) * (image.shape[0] / len(r_pair))  # 动态计算文本y轴位置
        ax.text(image.shape[1] + 10, text_y, f"{pair}: {score:.2f}", fontsize=12, color='black',
                verticalalignment='center')

    plt.show()


def main(args):
    save_model_path = os.path.join(f"{args.save_model_path}", f"{args.dataset}")
    model_file = os.path.join(save_model_path, "best_macro_f1.pth")

    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    embeddings_path = os.path.join(dataset_dir, "embedding", f"{args.data_type}_embeddings.npy")
    json_path = os.path.join(dataset_dir, f"{args.data_type}_detect.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerSG(args.feature_dim, args.resnet_dim, args.n_heads, args.num_encoder_layers,
                          args.dim_feedforward, args.num_classes, args.d_k, args.d_v).to(device)

    model.load_state_dict(torch.load(model_file))
    model = model.to(device)

    # filename:单独分析的图片。
    analyze_contributions_by_filename(model, '2017_67507096', embeddings_path, json_path, device)

if __name__ == "__main__":
    start_time = time.time()
    main(args)
    end_time = time.time()  # 记录结束时间
    print(f"程序运行时间: {end_time - start_time:.4f} 秒")