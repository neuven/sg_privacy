import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .layer_relprop import *



class FeatureExtraCombiner(nn.Module):
    def __init__(self, feature_dim, image_dim):
        super(FeatureExtraCombiner, self).__init__()
        self.W1 = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W2 = nn.Linear(image_dim, feature_dim, bias=False)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, combined_features, extra_matrix):
        combined = self.W1(combined_features) + self.W2(extra_matrix)
        combined = nn.ReLU()(combined)
        return self.dropout(combined)



#计算注意力信息、残差和归一化
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.matmul1 = MatMul()
        self.matmul2 = MatMul()
        self.add = Add()
        self.attn_gradients = None
        self.dropout = nn.Dropout(p=0.5)

    def save_attn_gradients(self, attn_gradients):
        # print("Gradient Hook Called")
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, Q, K, V, attn_mask):                             # Q: [batch_size, n_heads, len_q, d_k]
                                                                       # K: [batch_size, n_heads, len_k, d_k]
                                                                       # V: [batch_size, n_heads, len_v(=len_k), d_v]
                                                                       # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        scores = self.matmul1([Q, K.transpose(-1, -2)])
        scores = scores/np.sqrt(self.d_k)   # scores : [batch_size, n_heads, len_q, len_k]
        # scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # 将 mask 转换成布尔类型的张量，其中 0 对应 True，非 0 对应 False
        attention_mask = (attn_mask - 1) * 1e9  # 0---1e9; 1--0
        # bool_mask = attn_mask == 0
        # scores.masked_fill_(bool_mask, -1e9)                           # 如果是停用词P就等于 0
        scores = self.add([scores, attention_mask])
        attn = nn.Softmax(dim=-1)(scores)
        attn.retain_grad()
        attn.register_hook(self.save_attn_gradients)
        attn = self.dropout(attn)
        # context = torch.matmul(attn, V)                                # [batch_size, n_heads, len_q, d_v]
        context = self.matmul2([attn, V])
        return context, attn

    def relprop(self, cam, **kwargs):
        (cam1_0, cam2) = self.matmul2.relprop(cam, **kwargs)
        cam1_0 /= 2
        cam2 /= 2
        # attention mask
        (cam1, _) = self.add.relprop(cam1_0, **kwargs)

        # [query_layer, key_layer.transpose(-1, -2)]
        (cam1_1, cam1_2) = self.matmul1.relprop(cam1, **kwargs)
        cam1_1 /= 2
        cam1_2 /= 2
        return cam1_0, cam2, cam1_1, cam1_2


#多头注意力层
class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim, n_heads, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = Linear(feature_dim, d_k * n_heads, bias=False)
        self.W_K = Linear(feature_dim, d_k * n_heads, bias=False)
        self.W_V = Linear(feature_dim, d_v * n_heads, bias=False)
        self.fc = Linear(n_heads * d_v, feature_dim, bias=False)
        self.attention = ScaledDotProductAttention(d_k)
        self.n_heads = n_heads
        self.attention_head_size = int(feature_dim / n_heads)
        self.d_k = d_k
        self.d_v = d_v
        self.feature_dim = feature_dim
        self.dropout = nn.Dropout(p=0.5)
        self.attn_cam = None
        self.attn = None
        self.attn_gradients = None
        self.attn_hook = None

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_relprop(self, x):
        return x.permute(0, 2, 1, 3).flatten(2)

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def save_attn_gradients(self, attn_gradients):
        # print("Gradient Hook Called")
        # print(attn_gradients)
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, feature_dim]
        input_K: [batch_size, len_k, feature_dim]
        input_V: [batch_size, len_v(=len_k), feature_dim]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len] 扩充注意力mask

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = self.attention(Q, K, V, attn_mask)

        self.save_attn(attn)
        # print(attn)
        if self.attn_hook is not None:
            self.attn_hook.remove()
        attn.retain_grad()
        self.attn_hook =attn.register_hook(self.save_attn_gradients)
        attn1 = attn

        context = context.transpose(1, 2).reshape(batch_size, -1,self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.dropout(output)
        # LayerNorm和残差连接
        return nn.LayerNorm(self.feature_dim).cuda()(output + residual), attn1


    def relprop(self, cam, **kwargs):
        # layernorm pass; dropout pass
        cam = self.fc.relprop(cam, **kwargs)
        # Assume output_attentions == False
        cam = self.transpose_for_scores(cam) # [batch_size, n_heads, seq_len, attention_head_size=d_k]
        # [attention_probs, value_layer]
        cam1, cam2, cam1_1, cam1_2 = self.attention.relprop(cam, **kwargs)
        self.save_attn_cam(cam1)
        # query
        cam1_1 = self.transpose_for_scores_relprop(cam1_1)
        cam1_1 = self.W_Q.relprop(cam1_1, **kwargs)

        # key
        cam1_2 = self.transpose_for_scores_relprop(cam1_2.transpose(-1, -2))
        cam1_2 = self.W_K.relprop(cam1_2, **kwargs)

        # value
        cam2 = self.transpose_for_scores_relprop(cam2)
        cam2 = self.W_V.relprop(cam2, **kwargs)

        # cam = self.clone.relprop((cam1_1, cam1_2, cam2), **kwargs)
        return (cam1_1, cam1_2, cam2)





#前馈神经网络：两个全连接层，残差，LayerNorm归一化
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, feature_dim, dim_feedforward):
        super(PoswiseFeedForwardNet, self).__init__()
        self.feature_dim = feature_dim
        self.fc1 = Linear(feature_dim, dim_feedforward, bias=False)
        self.fc2 = Linear(dim_feedforward, feature_dim, bias=False)
        self.fc = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            nn.Dropout(p=0.5),
            self.fc2,
            nn.Dropout(p=0.5)
        )
        self.add = Add()
        self.clone = Clone()

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual, inputs = self.clone(inputs, 2)
        output = self.fc(inputs)
        add = self.add([output, residual])
        return nn.LayerNorm(self.feature_dim).cuda()(add)  # [batch_size, seq_len, d_model]

    def relprop(self, cam, **kwargs):
        # [hidden_states, input_tensor]
        (cam1, cam2)= self.add.relprop(cam, **kwargs)
        cam1 = self.fc2.relprop(cam1, **kwargs)
        cam1 = self.fc1.relprop(cam1, **kwargs)
        cam = self.clone.relprop((cam1, cam2), **kwargs)
        return cam





class EncoderLayer(nn.Module):
    def __init__(self, feature_dim, n_heads, d_k, d_v, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(feature_dim, n_heads, d_k, d_v)  #多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet(feature_dim, dim_feedforward)    #前馈神经网络
        self.dropout = nn.Dropout(p=0.5)
        self.clone = Clone()

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        h1, h2, h3 = self.clone(enc_inputs, 3)
        enc_outputs, attn = self.enc_self_attn(h1, h2, h3, enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.dropout(enc_outputs)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

    def relprop(self, cam, **kwargs):
        cam = self.pos_ffn.relprop(cam, **kwargs) # 分别为output和残差的cam
        (cam1, cam2, cam3) = self.enc_self_attn.relprop(cam, **kwargs)
        cam = self.clone.relprop((cam1, cam2, cam3), **kwargs)
        return cam


#整体encoder
class Encoder(nn.Module):
    def __init__(self, feature_dim, image_dim, n_heads, num_encoder_layers, d_k, d_v, dim_feedforward):
        super(Encoder, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #self.pos_emb = PositionalEncoding(d_model)
        self.feature_extra_combiner = FeatureExtraCombiner(feature_dim, image_dim)
        self.layers = nn.ModuleList([EncoderLayer(feature_dim, n_heads, d_k, d_v, dim_feedforward) for _ in range(num_encoder_layers)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, enc_inputs, attn_mask, extra_matrix):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        attn_mask: [batch_size, src_len, src_len]
        '''
        #enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        #enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_outputs = self.feature_extra_combiner(enc_inputs, extra_matrix) #融合图像特征 [batch_size, src_len, feature_dim]
        enc_outputs = self.dropout(enc_outputs)
        # # 消融实验：只保留前10个 token，其余部分设为 0
        # enc_outputs[:, 10:, :] = 0

        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs


#后半部分的encoder
class Encoder2(nn.Module):
    def __init__(self, feature_dim, n_heads, num_encoder_layers, d_k, d_v, dim_feedforward):
        super(Encoder2, self).__init__()
        #self.src_emb = nn.Embedding(src_vocab_size, d_model)
        #self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(feature_dim, n_heads, d_k, d_v, dim_feedforward) for _ in range(num_encoder_layers)])
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, enc_inputs, attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        attn_mask: [batch_size, src_len, src_len]
        '''
        #enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        #enc_outputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attns = []
        n = 1
        for layer in self.layers:
            # print("层数-------------------------------------" + str(n))
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_inputs, attn_mask)
            enc_self_attns.append(enc_self_attn)
            n += 1
        enc_outputs = self.dropout(enc_outputs)
        return enc_outputs, enc_self_attns

    def relprop(self, cam, **kwargs):
        for layer_module in reversed(self.layers):
            cam = layer_module.relprop(cam, **kwargs)
        return cam


#tranformer模型，子图生成版
class TransformerSG(nn.Module):
    def __init__(self, feature_dim, image_dim, n_heads, num_encoder_layers, dim_feedforward, num_classes, d_k, d_v):
        super(TransformerSG, self).__init__()
        self.encoder1 = Encoder(feature_dim, image_dim, n_heads, num_encoder_layers, d_k, d_v, dim_feedforward)
        self.encoder2 = Encoder2(feature_dim, n_heads, num_encoder_layers, d_k, d_v, dim_feedforward)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.fc = Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim

    def forward(self, x, attn_mask, subgraph_indices, extra_matrix):
        # 输入形状为 (batch_size, seq_length, feature_dim)，输出形状为(batch_size, seq_length, feature_dim)
        # encoder_outputs = self.encoder1(x, attn_mask, extra_matrix)
        encoder_outputs = self.encoder1(x, attn_mask, extra_matrix)

        # 对特征进行子图索引平均池化
        batch_size = encoder_outputs.size(0)
        pooled_features = []
        subgraph_counts = []

        for i in range(batch_size):
            batch_pooled_features = []
            for subgraph in subgraph_indices[i]:
                subgraph_features = encoder_outputs[i, subgraph, :]  # 提取子图特征
                #print("一个数据的一个子图")
                #print(subgraph_features)
                pooled_feature = subgraph_features.mean(dim=0)       # 子图特征平均池化
                #print("池化后的子图")
                #print(pooled_feature)
                batch_pooled_features.append(pooled_feature)
            #print("一个数据的池化子图特征")
            #print(batch_pooled_features)
            pooled_features.append(torch.stack(batch_pooled_features))
            subgraph_counts.append(len(batch_pooled_features)) #一个batch中数据的的子图数量

            # #消融实验，不分解子图。
            # pooled_features.append(encoder_outputs[i])  # 直接使用 encoder_outputs
            # subgraph_counts.append(encoder_outputs[i].shape[0])  # 记录序列长度

        # 拼接所有池化特征
        max_subgraph_count = max(subgraph_counts)
        padded_pooled_features = []
        mask = []

        for i, features in enumerate(pooled_features):
            padding_length = max_subgraph_count - len(features)
            if len(features) < max_subgraph_count:
                padding = torch.zeros((padding_length, self.feature_dim)).to(features.device)
                features = torch.cat([features, padding], dim=0)
            padded_pooled_features.append(features)
            # 创建mask, 填充的地方为0, 其他地方为1
            feature_mask = torch.cat([torch.ones(len(features) - padding_length), torch.zeros(padding_length)]).to(features.device)
            mask.append(feature_mask)


        # print("一个batch的子图特征为：")
        # print(padded_pooled_features)


        padded_pooled_features = torch.stack(padded_pooled_features)  # [batch_size, max_num_subgraphs, feature_dim]
        mask = torch.stack(mask)  # [batch_size, max_num_subgraphs]

        # 在开头添加 cls_token
        batch_size, seq_len, _ = padded_pooled_features.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        padded_pooled_features = torch.cat((cls_tokens, padded_pooled_features), dim=1)
        # 更新mask, cls_token的地方为1
        cls_mask = torch.ones(batch_size, 1).to(mask.device)
        mask = torch.cat((cls_mask, mask), dim=1)
        # 生成全局mask
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [batch_size, seq_len + 1, seq_len + 1]
        # print("一个batch的子图特征mask：")
        # print(attn_mask)

        # 传递给新的 encoder
        outputs, attn_weights = self.encoder2(padded_pooled_features, attn_mask)
        # 提取 cls_token 的输出,全连接层进行分类
        cls_output = outputs[:, 0]
        output = self.fc(cls_output)


        # 全连接层进行分类
        #output = self.fc(padded_pooled_features.mean(dim=1))  # 平均池化后再分类
        return output, attn_weights

    def relprop(self, cam, **kwargs):
        cam = self.fc.relprop(cam, **kwargs)
        cam = self.encoder2.relprop(cam, **kwargs)
        return cam


