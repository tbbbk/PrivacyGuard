import torch
import torch.nn as nn


class FFM(nn.Module):
    def __init__(self, embedding_dim):
        super(FFM, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, node_embed, edge_index, edge_embed):
        num_nodes, _ = node_embed.size()
        num_edges = edge_index.size(1)

        # 初始化节点的输出嵌入
        output_embed = node_embed.clone()

        for i in range(num_edges):
            src_node = edge_index[0, i]
            tgt_node = edge_index[1, i]
            connection_embed = edge_embed[i].unsqueeze(0)
            ffm_interaction = node_embed[src_node] * node_embed[tgt_node] * connection_embed
            output_embed[src_node] += ffm_interaction

        return output_embed


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# 示例数据
embedding_dim = 32  # 嵌入维度
num_nodes = 10  # 节点数量
num_edges = 10  # 连接数量

node_embed = torch.randn(num_nodes, embedding_dim)  # 节点嵌入
edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]], dtype=torch.long)  # 连接信息
edge_embed = torch.randn(num_edges, embedding_dim)  # 连接内容的嵌入

# 创建FFM模型
ffm_model = FFM(embedding_dim)

# 运行FFM模型
output_embed = ffm_model(node_embed, edge_index, edge_embed)

# 创建MLP模型
mlp_model = MLP(embedding_dim, [64, 32], 1)

# 运行MLP模型
output_probabilities = mlp_model(output_embed).squeeze()

print("输出节点嵌入：", output_embed)
print("输出节点可能性：", output_probabilities)