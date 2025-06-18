import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

"""
生成模拟数据用于测试
"""

def generate_sample_data(num_graphs, feature_dim, image_dim):
    # 生成num_graphs个场景图数据
    graphs_data = []

    for _ in range(num_graphs):
        # 生成节点数量和关系数量
        num_nodes = np.random.randint(5, 10)  # 节点数量随机选择5到9个
        num_relations = np.random.randint(1, 5)  # 随机选择1到4个关系

        # 生成节点特征
        node_features = np.random.randn(num_nodes, feature_dim)

        # 生成关系特征
        relation_features = np.random.randn(num_relations, feature_dim)

        #生成工图像特征矩阵
        image_features = np.ones((num_nodes, image_dim))

        # 生成领接矩阵
        adjacency_matrix = np.zeros((num_nodes + num_relations, num_nodes + num_relations))
        # 生成关系之间的连接关系
        for i in range(num_relations):
            for j in range(i + 1, num_relations):
                if np.random.rand() < 0.5:  # 随机连接关系
                    adjacency_matrix[num_nodes + i, num_nodes + j] = 1
                    adjacency_matrix[num_nodes + j, num_nodes + i] = 1
        # 对角线设为1
        np.fill_diagonal(adjacency_matrix, 1)

        # 生成子图索引
        num_subgraphs = np.random.randint(1, 4)  # 随机生成1到3个子图
        subgraph_indices = []
        for _ in range(num_subgraphs):
            subgraph_size = np.random.randint(3, 4)  # 每个子图包含3个节点或关系
            indices = np.random.choice(num_nodes, subgraph_size, replace=False)
            subgraph_indices.append(indices.tolist())

        # 生成标签
        label = np.random.randint(0, 2)

        # 将生成的数据存储到列表中
        graphs_data.append({
            'filename': "sample",
            'n_embeddings': node_features,
            'r_embeddings': relation_features,
            'cropped_embeddings': image_features,
            'adjacency_matrix': adjacency_matrix,
            'r_pair': subgraph_indices,
            'label': label
        })
    return graphs_data


def generate_adjacency(p_matrix, graph_index, num_node):
    for subgraph in graph_index:
        p_matrix[subgraph[0], subgraph[1]] = 1
        p_matrix[subgraph[1], subgraph[0]] = 1
        p_matrix[subgraph[0], subgraph[2]] = 1
        p_matrix[subgraph[2], subgraph[0]] = 1
        p_matrix[subgraph[1], subgraph[2]] = 1
        p_matrix[subgraph[2], subgraph[1]] = 1
    # 第一行和第一列统一标1,表示场景部分和其他部分都有关系
    p_matrix[0, :num_node] = 1
    p_matrix[:num_node, 0] = 1
    # 对角线统一标1
    np.fill_diagonal(p_matrix, 1)
    return p_matrix


def print_graph_data(g_data):
    # 打印得到的graph数据
    for idx, data in enumerate(g_data):
        print(f"图片:")
        print(data['filename'])
        print("节点特征：")
        print(data['n_embeddings'])
        print("\n关系特征：")
        print(data['r_embeddings'])
        print("\n领接矩阵：")
        print(data['adjacency_matrix'])
        print("\n子图索引：", data['r_pair'])
        print("\n标签：", data['label'])
        print("\n")

class SceneGraphDataset(Dataset):
    def __init__(self, graphs_data):
        self.graphs_data = graphs_data

    def __len__(self):
        return len(self.graphs_data)

    def __getitem__(self, idx):
        graph_data = self.graphs_data[idx]

        # 获取节点特征,关系特征,子图索引，图像特征
        file_name = graph_data['filename']
        node_features = np.array(graph_data['n_embeddings'])
        relation_features = np.array(graph_data['r_embeddings'])
        subgraph_indices = np.array(graph_data['r_pair'])
        image_features = np.array(graph_data['cropped_embeddings'])

        # 判断relation_features是否为空，如果为空则直接使用node_features
        if relation_features.size == 0:
            combined_features = node_features
        else:
            # 合并节点特征和关系特征
            combined_features = np.concatenate([node_features, relation_features], axis=0)

        # 扩充图像特征矩阵
        padding_length = combined_features.shape[0] - image_features.shape[0]
        extra_padding = np.zeros((padding_length, image_features.shape[1]))
        extra_matrix = np.concatenate([image_features, extra_padding], axis=0)

        # 修改子图索引
        node_offset = node_features.shape[0]
        max_index = combined_features.shape[0] - 1

        modified_subgraph_indices = []
        if subgraph_indices.size == 0:
            modified_subgraph_indices = [[0,0,0]]
        else:
            for subgraph in subgraph_indices:
                modified_subgraph = subgraph.copy()  # 复制当前子图索引
                modified_subgraph[2] += node_offset  # 加上节点偏移量
                if modified_subgraph[2] > max_index:
                    modified_subgraph[2] = max_index  # 如果超过最大索引，设置为最大索引
                modified_subgraph_indices.append(modified_subgraph)

        # 获取邻接矩阵和标签
        if file_name == "sample":
            final_adjacency = graph_data['adjacency_matrix']
        else:
            num_features = combined_features.shape[0]
            adjacency_matrix = np.zeros((num_features, num_features), dtype=np.float32)
            final_adjacency = generate_adjacency(adjacency_matrix,modified_subgraph_indices,node_features.shape[0])

        label = graph_data['label']

        return combined_features, final_adjacency, modified_subgraph_indices, label, extra_matrix

def collate_fn(batch):
    # 获取本 batch 中节点+关系数量的最大值
    max_num_features = max(x[0].shape[0] for x in batch)
    feature_dim = batch[0][0].shape[1]  # 特征维度

    padded_features = []
    padded_adjacency_matrices = []
    subgraph_indices_list = []
    labels = []
    padded_extra_matrices = []

    for features, adjacency_matrix, subgraph_indices, label, extra_matrix in batch:
        num_features = features.shape[0]

        # 填充特征
        padded_features.append(
            np.pad(features, ((0, max_num_features - num_features), (0, 0)), mode='constant', constant_values=0))

        # 填充邻接矩阵
        padded_adj_matrix = np.zeros((max_num_features, max_num_features), dtype=np.float32)
        padded_adj_matrix[:num_features, :num_features] = adjacency_matrix

        padded_adjacency_matrices.append(padded_adj_matrix)
        subgraph_indices_list.append(subgraph_indices)
        labels.append(label)
        # 填充图像特征矩阵
        padded_extra_matrix = np.pad(extra_matrix, ((0, max_num_features - num_features), (0, 0)), mode='constant', constant_values=0)
        padded_extra_matrices.append(padded_extra_matrix)

    padded_features = torch.tensor(np.stack(padded_features), dtype=torch.float32)
    padded_adjacency_matrices = torch.tensor(np.stack(padded_adjacency_matrices), dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long).view(-1)
    padded_extra_matrices = torch.tensor(np.stack(padded_extra_matrices), dtype=torch.float32)

    return padded_features, padded_adjacency_matrices, labels, subgraph_indices_list, padded_extra_matrices


if __name__ == "__main__":
    sample_graph_data = generate_sample_data(10)
    # 打印生成的数据
    # print_graph_data(sample_graph_data)

    # 创建数据集实例
    dataset = SceneGraphDataset(sample_graph_data)

    # 创建数据加载器，使用自定义的 collate 函数
    batch_size = 2
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # 示例用法：遍历数据加载器
    for combined_features, adjacency_matrix, label, subgraph_indices_list, extra_matrix in data_loader:
        # 此处可以将数据输入到模型中进行训练
        print("特征：", combined_features.shape, combined_features)
        print("邻接矩阵：", adjacency_matrix.shape, adjacency_matrix)
        print("标签：", label)
        print("子图索引：", subgraph_indices_list)
        print("图像特征矩阵：", extra_matrix.shape, extra_matrix)
        # break  # 仅打印第一个batch的数据