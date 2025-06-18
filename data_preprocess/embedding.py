import json
from transformers import BertTokenizer, BertModel
import torch
import torchvision
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from main import parse_arguments


"""
第五步：将检测的文字和图片向量化（存储在json文件中），向量化后存储在npy文件中。
"""

class InvalidPositionError(Exception):
    """Custom exception for invalid image crop positions."""
    pass


def text_to_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()


def process_image(i_path, positions):
    image = cv2.imread(i_path)
    if positions == ['0', '0', '0', '0']:
        crop_image = image
    else:
        x1, y1, x2, y2 = [int(float(pos)) for pos in positions]
        # 修正无效坐标
        height, width, _ = image.shape
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        crop_position = [x1,y1,x2,y2]
        # 检查修正后的坐标是否有效
        if x1 >= x2 or y1 >= y2:
            print(f"Invalid crop positions: {crop_position}. Outputting a black image.")
            # raise InvalidPositionError(f"Invalid crop positions: {crop_position}")
            crop_image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            crop_image = image[y1:y2, x1:x2]
    resized_image = cv2.resize(crop_image, (224, 224))
    return resized_image


def image_to_embedding(image):
    # 使用预训练的图像嵌入模型
    # 这里以ResNet为例
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet.eval()
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        i_embedding = resnet(image_tensor)
    return i_embedding.squeeze().numpy()


def visualize_and_save_cropped_images(cropped_images_v, n_list_v, v_filename, output_folder):
    num_images = len(cropped_images_v)
    # 计算行数和列数
    num_cols = 5  # 每行显示5张图片
    num_rows = (num_images + num_cols - 1) // num_cols  # 向上取整计算行数

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
    axes = axes.flatten()  # 将二维数组展开为一维数组

    for ax, img, label in zip(axes, cropped_images_v, n_list_v):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(label, fontsize=10)
        ax.axis('off')

    # 隐藏多余的子图
    for ax in axes[num_images:]:
        ax.axis('off')

    plt.suptitle(f"File: {v_filename}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整布局以防止标题被裁剪

    # 保存图像
    output_file = os.path.join(output_folder, f"{v_filename}_visualization.png")
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def main(args):
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    json_input_path = os.path.join(dataset_dir, f"{args.data_type}_detect.json")
    image_path = os.path.join(dataset_dir, f"images/{args.data_type}")
    visualize_dir = os.path.join(image_path, "annotate")  # 用于保存可视化图像的文件夹
    embedding_output_file = os.path.join(dataset_dir, f"embedding/{args.data_type}_embeddings.npy")  # 用于保存嵌入向量的文件
    bert_model_path = f"{args.bert_path}"

    # 创建保存embedding的文件夹（如果不存在）
    os.makedirs(os.path.join(dataset_dir, "embedding"), exist_ok=True)

    # 初始化BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    model = BertModel.from_pretrained(bert_model_path)

    with open(json_input_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)


    # 如果npy文件已经存在，加载现有的嵌入
    if os.path.exists(embedding_output_file):
        existing_embeddings = np.load(embedding_output_file, allow_pickle=True).tolist()
        existing_filenames = {entry["filename"] for entry in existing_embeddings}
    else:
        existing_embeddings = []
        existing_filenames = set()

    batch_count = 0  # 用于记录批次
    all_embeddings = []
    for data_entry in tqdm(data_list, desc="embedding image files"):
        filename = data_entry["filename"]
        n_list = data_entry["n_list"]
        r_list = data_entry["r_list"]
        n_position = data_entry["n_position"]
        r_pair = data_entry["r_pair"]
        label = data_entry["label"]

        #已经存在数据，跳过
        if filename in existing_filenames:
            print(f"Skipping {filename} as it already exists in the embeddings.")
            continue
        batch_count += len(existing_embeddings//1000)
        existing_embeddings = []

        # 嵌入节点名称和关系名称
        n_embeddings = [text_to_embedding(n, tokenizer, model) for n in n_list]
        r_embeddings = [text_to_embedding(r, tokenizer, model) for r in r_list]

        # 裁剪图片并嵌入
        image_file = os.path.join(image_path, f"{filename}.jpg")
        cropped_embeddings = []
        cropped_images = []
        for idx, pos in enumerate(n_position):
            if idx != 0 and pos == ['0', '0', '0', '0']:
                cropped_embeddings.append(np.zeros(1000))
                cropped_images.append(np.zeros((224, 224, 3), dtype=np.uint8))  # 假设使用全黑图像代替无效区域
            else:
                try:
                    cropped_image = process_image(image_file, pos)
                    embedding = image_to_embedding(cropped_image)
                    cropped_embeddings.append(embedding)
                    cropped_images.append(cropped_image)
                except Exception as e:
                    print(f"处理图片时出错: {e}")
                    cropped_embeddings.append(np.zeros(1000))
                    cropped_images.append(np.zeros((224, 224, 3), dtype=np.uint8))

        # 可视化并保存裁剪后的图像
        # visualize_and_save_cropped_images(cropped_images, n_list, filename, visualize_dir)

        # 将所有嵌入和其他信息存储到一个字典中
        entry_embeddings = {
            "filename": filename,
            "r_embeddings": r_embeddings,
            "n_embeddings": n_embeddings,
            "cropped_embeddings": cropped_embeddings,
            "r_pair": r_pair,
            "label": label
        }
        existing_embeddings.append(entry_embeddings)

        # 输出结果
        print("文件名:", filename)
        print("关系名称嵌入维度:", [e.shape for e in r_embeddings])   #768
        print("节点名称嵌入维度:", [e.shape for e in n_embeddings])   #768
        print("裁剪后图片嵌入维度:", [e.shape for e in cropped_embeddings])  #1000
        print("关系对应的节点顺序:", r_pair)

        # 每处理一个文件，就检查是否达到批次大小。并产生一个合并文件
        if len(existing_embeddings) >= 1000:
            batch_count += 1
            batch_file = f"{embedding_output_file.rsplit('.', 1)[0]}_{batch_count}.npy"
            np.save(batch_file, existing_embeddings)
            print(f"Saved batch {batch_count} to {batch_file}")
            existing_embeddings = []  # 清空列表

            final_embeddings = []
            for i in range(1, batch_count + 1):
                batch_file = f"{embedding_output_file.rsplit('.', 1)[0]}_{i}.npy"
                if os.path.exists(batch_file):
                    batch_data = np.load(batch_file, allow_pickle=True)
                    final_embeddings.extend(batch_data)
            # 保存合并后的完整npy文件
            np.save(embedding_output_file, final_embeddings)
            print(f"All batches have been merged and saved to {embedding_output_file}")


        # # 每处理一个文件，就保存到npy文件
        # np.save(embedding_output_file, existing_embeddings)
    # 处理完成后保存剩余的嵌入
    if existing_embeddings:
        batch_count += 1
        batch_file = f"{embedding_output_file.rsplit('.', 1)[0]}_{batch_count}.npy"
        np.save(batch_file, existing_embeddings)
        print(f"Saved final batch {batch_count} to {batch_file}")
        final_embeddings = []
        for i in range(1, batch_count + 1):
            batch_file = f"{embedding_output_file.rsplit('.', 1)[0]}_{i}.npy"
            if os.path.exists(batch_file):
                batch_data = np.load(batch_file, allow_pickle=True)
                final_embeddings.extend(batch_data)
        # 保存合并后的完整npy文件
        np.save(embedding_output_file, final_embeddings)
        print(f"All batches have been merged and saved to {embedding_output_file}")

    # # 保存所有嵌入向量到一个本地文件
    # np.save(embedding_output_file, all_embeddings)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)