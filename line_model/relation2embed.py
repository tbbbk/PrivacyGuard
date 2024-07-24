import torch
import torch.nn as nn
import yaml
import warnings
class RelationEmbedding(nn.Module):
    def __init__(self, num_relations, embedding_dim):
        super(RelationEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_relations, embedding_dim)
        self.num_relations = num_relations
    def forward(self, relation, to="nn_embed"):
        # relation: LongTensor of shape (batch_size,)
        relation = relation.long()
        if(to == "nn_embed"):
            embedded = self.embedding(relation)
        elif(to == "onehot"):
            embedded = torch.nn.functional.one_hot(relation, num_classes=self.num_relations)
        else:
            warnings.warn('err, please check the embeding or onehot', DeprecationWarning)
        return embedded



if __name__ == "__main__":
    with open('./configs/model.yaml') as f:
        model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"]
    # 设置随机种子
    torch.manual_seed(seeds_num)
    # 示例数据
    num_relations = model_info["model"]["relationship"]["num_relations"]  # 类别的数量
    embedding_dim = model_info["model"]["relationship"]["embedding_dim"]  # 嵌入维度
    relation = torch.tensor([2, 4, 1], dtype=torch.long)  # 输入的关系

    # 创建关系嵌入模型
    embedding_model = RelationEmbedding(num_relations, embedding_dim)

    # 转换关系为嵌入
    embed = embedding_model(relation)
    print("嵌入结果：", embed)