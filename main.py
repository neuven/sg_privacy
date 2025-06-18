import argparse
import os.path
import sys

from data_preprocess.dataset import generate_sample_data, SceneGraphDataset, DataLoader, collate_fn, print_graph_data
from model.SG_Transformer import TransformerSG
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
from sg_utils.sg_util import load_data_from_npy, print_loader, test_model

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and visualize image embeddings.")
    # parser.add_argument("--data_path",type=str, default='F:/data', help='存放数据的位置--本地用')
    parser.add_argument("--data_path", type=str, default='/home/liangxy/pycharm/sg_privacy/data', help='存放数据的位置--服务器用')
    parser.add_argument("--dataset", type=str, default='privacy_alert_v2', choices=['VISPR', 'picalert','privacy_alert_v2'],
                        help="Name of the dataset.")
    parser.add_argument("--data_type", type=str, default='val', choices=['train', 'val', 'test'],
                        help="Type of the data (train, val, test).")
    parser.add_argument("--run_mode", type=str, default='remain',choices=['split','remain','sample'],
                        help="数据读取模式")
    parser.add_argument("--device_id", type=int, default=1, help='gpu id')
    # parser.add_argument("--bert_path", type=str, default='F:/data/models/bert_base_uncased', help='存放bert预训练模型的位置')
    parser.add_argument("--bert_path", type=str, default='/home/liangxy/pycharm/pretrained_model/bert_base_uncased',
                        help='存放bert预训练模型的位置')
    parser.add_argument("--save_model_path", type=str, default="/home/liangxy/pycharm/sg_privacy/ckpt")

    #transformer
    parser.add_argument("--feature_dim", type=int, default=768, help='Embedding 的维度')
    parser.add_argument("--resnet_dim", type=int, default=1000, help='图像特征Embedding 的维度')
    parser.add_argument("--n_heads", type=int, default=8, help='Multi-Head Attention多头个数')
    parser.add_argument("--num_encoder_layers", type=int, default=4, help='encoder层数')
    parser.add_argument("--dim_feedforward", type=int, default=1536, help='前向传播隐藏层维度')
    parser.add_argument("--num_classes", type=int, default=2, help='最终隐私分类public/private')
    parser.add_argument("--d_k", type=int, default=96, help='K(=Q)的维度,一般，d_k = d_v = feature_dim // n_heads')
    parser.add_argument("--d_v", type=int, default=96, help='v的维度')
    parser.add_argument("--num_epochs", type=int, default=30, help='训练轮数')
    parser.add_argument("--batch_size", type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='优化率')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")

    if args.run_mode == "sample":
        graph_data = generate_sample_data(20, args.feature_dim, args.resnet_dim)
    elif args.run_mode == "split":
        npy_file_path = os.path.join(dataset_dir, f"embedding/{args.data_type}_embeddings.npy")
        graph_data = load_data_from_npy(npy_file_path)
        # print_graph_data(graph_data)            # 打印生成的数据

        # 创建数据集实例
        full_dataset = SceneGraphDataset(graph_data)

        # 定义划分比例
        train_ratio = 0.8
        train_size = int(train_ratio * len(full_dataset))
        val_size = len(full_dataset) - train_size
        # 划分数据集
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    else:
        train_npy = os.path.join(dataset_dir, f"embedding/train_embeddings.npy")
        val_npy = os.path.join(dataset_dir, f"embedding/val_embeddings.npy")
        train_graph_data = load_data_from_npy(train_npy)
        val_graph_data = load_data_from_npy(val_npy)
        train_dataset = SceneGraphDataset(train_graph_data)
        val_dataset = SceneGraphDataset(val_graph_data)


    # 创建数据加载器，使用自定义的 collate 函数
    data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print_loader(data_loader)   # 示例print:遍历数据加载器

    # 创建模型
    torch.cuda.set_device(args.device_id)
    model = TransformerSG(args.feature_dim, args.resnet_dim, args.n_heads, args.num_encoder_layers, args.dim_feedforward, args.num_classes, args.d_k, args.d_v)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 定义学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # 初始化变量以跟踪最佳模型
    best_acc = 0
    best_pub_prec = 0
    best_pub_rec = 0
    best_priv_prec = 0
    best_priv_rec = 0
    best_macro_f1 = 0
    save_model_path = os.path.join(f"{args.save_model_path}", f"{args.dataset}/ablation")
    os.makedirs(save_model_path, exist_ok=True)

    print("----------------------------start training-----------------------------------------------")
    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, (features, adjacency_matrix, labels, subgraph_indices_list, padded_extra_matrices) in enumerate(
                data_loader):
            features, adjacency_matrix, labels, extra_matrix = features.to(device), adjacency_matrix.to(
                device), labels.to(device), padded_extra_matrices.to(device)

            # 前向传播
            outputs, attn_weights = model(features, adjacency_matrix, subgraph_indices_list, extra_matrix)
            # print(labels)
            # print(outputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            # 进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            if (batch_idx + 1) % 1 == 0:
                sys.stdout.write(
                    f"\rEpoch [{epoch + 1}/{args.num_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")
                sys.stdout.flush()
                # print(
                #     f"Epoch [{epoch + 1}/{args.num_epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.4f}")

        print("-"*50)
        # 在每个epoch结束时进行验证
        print("训练集效果：")
        test_model(model, data_loader, device)
        print("测试集效果：")
        avg_loss, acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = test_model(model, val_loader, device)
        scheduler.step(avg_loss)

        # 保存效果最好的模型
        if acc > best_acc:
            best_acc = acc
            best_model_path = os.path.join(save_model_path, "best_acc.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with accuracy: {best_acc:.4f}")

        if pub_prec > best_pub_prec:
            best_pub_prec = pub_prec
            best_model_path = os.path.join(save_model_path, "best_pub_prec.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with public precision: {best_pub_prec:.4f}")

        if pub_rec > best_pub_rec:
            best_pub_rec = pub_rec
            best_model_path = os.path.join(save_model_path, "best_pub_rec.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with public recall: {best_pub_rec:.4f}")

        if priv_prec > best_priv_prec:
            best_priv_prec = priv_prec
            best_model_path = os.path.join(save_model_path, "best_priv_prec.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with private precision: {best_priv_prec:.4f}")

        if priv_rec > best_priv_rec:
            best_priv_rec = priv_rec
            best_model_path = os.path.join(save_model_path, "best_priv_rec.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with private recall: {best_priv_rec:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_model_path = os.path.join(save_model_path, "best_macro_f1.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with macro F1 score: {best_macro_f1:.4f}")


    print("训练结束")

    # 测试模型
    print("最后一次epoch测试：")
    test_model(model, data_loader, device)