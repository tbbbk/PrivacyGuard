import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, dropout):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        self.W = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_heads)])
        self.a = nn.ModuleList([nn.Linear(2 * out_features, 1) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embed, connection_info):
        batch_size, num_nodes, _ = node_embed.size()

        # 多头注意力计算
        outputs = []
        for i in range(self.num_heads):
            h = self.W[i](node_embed)
            a_input = torch.cat([h.unsqueeze(2).expand(-1, -1, num_nodes, -1),
                                 h.unsqueeze(1).expand(-1, num_nodes, -1, -1)], dim=-1)
            e = torch.tanh(self.a[i](a_input)).squeeze(-1)
            attention = F.softmax(e, dim=2)
            attention = self.dropout(attention)

            # 加权求和
            h_prime = torch.matmul(attention, h)
            outputs.append(h_prime)

        # 拼接多头输出
        output_embed = torch.cat(outputs, dim=-1)

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
embedding_dim = 16  # 嵌入维度
num_nodes = 10  # 节点数量
num_heads = 4  # 多头注意力头数

node_embed = torch.randn(1, num_nodes, embedding_dim)  # 节点嵌入
connection_info = torch.randint(0, 2, (1, num_nodes, num_nodes, embedding_dim))  # 连接信息
edge_embed = torch.randn(1, num_nodes, num_nodes, embedding_dim)  # 连接的嵌入

# 创建GAT模型
gat_model = GATLayer(embedding_dim, embedding_dim, num_heads, dropout=0.2)

# 运行GAT模型
output_embed = gat_model(node_embed, connection_info * edge_embed)

# 创建MLP模型
mlp_model = MLP(embedding_dim * num_heads, [32, 16], 1)

# 运行MLP模型
output_probabilities = mlp_model(output_embed.squeeze())

print("输出节点嵌入：", output_embed)
print("输出节点可能性：", output_probabilities)