import argparse
import numpy as np
import torch
import glob

# compute rollout between attention layers
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    # adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
    """
    计算从某一层开始的注意力回滚，结合残差连接。
    参数:
    - all_layer_matrices: 所有层的注意力矩阵列表，每个矩阵形状为 (batch_size, num_tokens, num_tokens)。
    - start_layer: 开始计算回滚的层索引。
    返回:
    - joint_attention: 从 start_layer 开始回滚后的联合注意力矩阵。
    """
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                          for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer+1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


class Generator:
    def __init__(self, model):
        """
        初始化 Generator 类。

        参数:
        - model: 预训练的模型，用于生成解释。
        """
        self.model = model
        self.model.eval()  # 设置模型为评估模式，以禁用 dropout 等训练时行为

    def forward(self, input_ids, attention_mask):
        """
        执行前向传播。

        参数:
        - input_ids: 输入文本的 token ID。
        - attention_mask: 注意力掩码，标识输入中实际内容与填充的部分。

        返回:
        - 模型输出
        """
        return self.model(input_ids, attention_mask)

    def generate_LRP(self, data_loader, device, index=None, start_layer=0):
        """
        生成 LRP（Layer-wise Relevance Propagation）解释。
        返回:
        - 从起始层开始的注意力回滚结果。
        - rollouts[0][:, 0]: 第一个数据样本的注意力回滚结果。
        - predicts[0]: 第一个数据样本的预测类别。
        """
        # 得到结果
        rollouts = []
        predicts = []
        self.model.eval()
        for features, adjacency_matrix, labels, subgraph_indices_list, padded_extra_matrices in data_loader:
            features, adjacency_matrix, labels, extra_matrix = features.to(device), adjacency_matrix.to(device), labels.to(device), padded_extra_matrices.to(device)
            # 前向传播
            output, attn_weight = self.model(features, adjacency_matrix, subgraph_indices_list, extra_matrix)
            _, predicted = torch.max(output.data, 1)

            # output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            kwargs = {"alpha": 1}

            # 如果没有指定索引，则选择输出中概率最大的类别
            if index is None:
                index = np.argmax(output.cpu().data.numpy(), axis=-1)

            # 创建 one-hot 编码向量用于计算目标类别的梯度
            one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
            one_hot[0, index] = 1
            one_hot_vector = one_hot
            one_hot = torch.from_numpy(one_hot).requires_grad_(True)
            one_hot = torch.sum(one_hot.cuda() * output)

            # 计算梯度
            self.model.zero_grad()
            one_hot.backward(retain_graph=True)

            # 执行 LRP 相关性传播
            self.model.relprop(torch.tensor(one_hot_vector).to(device), **kwargs)

            # 收集每个块的注意力得分和梯度
            cams = []
            blocks = self.model.encoder2.layers
            # for blk in blocks:
            #     grad = blk.enc_self_attn.get_attn_gradients()
            #     cam = blk.enc_self_attn.get_attn_cam()
            #     cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            #     grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            #     cam = grad * cam  # 计算 Grad-CAM
            #     cam = cam.clamp(min=0).mean(dim=0)  # 取正值的平均
            #     cams.append(cam.unsqueeze(0))
            for blk in blocks:
                grad = blk.enc_self_attn.get_attn_gradients()
                cam = blk.enc_self_attn.get_attn_cam()
                if grad is not None:
                    cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                    grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                    cam = grad * cam  # 计算 Grad-CAM
                    cam = cam.clamp(min=0).mean(dim=0)  # 取正值的平均
                    cams.append(cam.unsqueeze(0))

            # 计算注意力回滚
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            rollout[:, 0, 0] = rollout[:, 0].min()  # 设置特殊 token 的注意力为最小值
            rollouts.append(rollout)
            predicts.extend(predicted.cpu().numpy())

        return rollouts[0][:, 0], predicts[0]

    def generate_LRP_last_layer(self, input_ids, attention_mask, index=None):
        """
        生成仅限于最后一层的 LRP 解释。

        参数:
        - input_ids: 输入文本的 token ID。
        - attention_mask: 注意力掩码。
        - index: 目标类别的索引。如果未提供，将使用预测的最大类别。

        返回:
        - 最后一层注意力得分。
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}
        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # 创建 one-hot 编码向量用于计算目标类别的梯度
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        # 计算梯度
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # 执行 LRP 相关性传播
        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        # 计算最后一层的注意力图
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn_cam()[0]
        cam = cam.clamp(min=0).mean(dim=0).unsqueeze(0)  # 聚合注意力得分
        cam[:, 0, 0] = 0  # 设置特殊 token 的注意力为 0
        return cam[:, 0]

    def generate_full_lrp(self, input_ids, attention_mask, index=None):
        """
        生成完整的 LRP 解释，包括所有层的相关性。

        参数:
        - input_ids: 输入文本的 token ID。
        - attention_mask: 注意力掩码。
        - index: 目标类别的索引。如果未提供，将使用预测的最大类别。

        返回:
        - 每个输入 token 的 LRP 相关性得分。
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index is None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        # 创建 one-hot 编码向量用于计算目标类别的梯度
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        # 计算梯度
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        # 执行 LRP 相关性传播并求和
        cam = self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)
        cam = cam.sum(dim=2)
        cam[:, 0] = 0  # 设置特殊 token 的相关性为 0
        return cam

    def generate_attn_last_layer(self, input_ids, attention_mask, index=None):
        """
        生成最后一层的注意力得分。

        参数:
        - input_ids: 输入文本的 token ID。
        - attention_mask: 注意力掩码。
        - index: 目标类别的索引（未使用，但保持一致性）。

        返回:
        - 最后一层平均注意力得分。
        """
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()[0]
        cam = cam.mean(dim=0).unsqueeze(0)  # 聚合多头注意力得分
        cam[:, 0, 0] = 0  # 设置特殊 token 的注意力为 0
        return cam[:, 0]

    def generate_rollout(self, input_ids, attention_mask, start_layer=0, index=None):
        """
        生成注意力回滚，从指定层开始。

        参数:
        - input_ids: 输入文本的 token ID。
        - attention_mask: 注意力掩码。
        - start_layer: 从哪个层开始计算注意力回滚。
        - index: 目标类别的索引（未使用，但保持一致性）。

        返回:
        - 从指定层开始的注意力回滚结果。
        """
        self.model.zero_grad()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        blocks = self.model.bert.encoder.layer
        all_layer_attentions = []

        # 收集所有层的注意力信息
        for blk in blocks:
            attn_heads = blk.attention.self.get_attn()
            avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
            all_layer_attentions.append(avg_heads)

        # 计算注意力回滚
        rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
        rollout[:, 0, 0] = 0  # 设置特殊 token 的注意力为 0
        return rollout[:, 0]

    def generate_attn_gradcam(self, input_ids, attention_mask, index=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
        kwargs = {"alpha": 1}

        if index == None:
            index = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, index] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        self.model.relprop(torch.tensor(one_hot_vector).to(input_ids.device), **kwargs)

        cam = self.model.bert.encoder.layer[-1].attention.self.get_attn()
        grad = self.model.bert.encoder.layer[-1].attention.self.get_attn_gradients()

        cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0).unsqueeze(0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam[:, 0, 0] = 0
        return cam[:, 0]

