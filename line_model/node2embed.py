import torch
import torch.nn as nn
import yaml
import warnings

class CategoryEmbedding(nn.Module):
    def __init__(self, num_categories, embedding_dim):
        super(CategoryEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.num_categories = num_categories
    def forward(self, category, to="nn_embed"):
        # category: LongTensor of shape (batch_size,)
        if(to == "nn_embed"):
            embedded = self.embedding(category)
        elif(to == "onehot"):
            embedded = torch.nn.functional.one_hot(category, num_classes=self.num_categories)
        else:
            # print('err, please check the embeding or onehot')
            warnings.warn('err, please check the embeding or onehot', DeprecationWarning)
        return embedded


if __name__ == "__main__":
    with open('./configs/model.yaml') as f:
        model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"]
    # 设置随机种子
    torch.manual_seed(seeds_num)
    # 示例数据
    num_categories = model_info["model"]["node"]["num_categories"]  # 类别的数量
    embedding_dim = model_info["model"]["node"]["embedding_dim"]  # 嵌入维度
    category = torch.tensor([2, 5, 1], dtype=torch.long)  # 输入的类别

    # 创建类别嵌入模型
    embedding_model = CategoryEmbedding(num_categories, embedding_dim)

    # 转换类别为嵌入
    embed = embedding_model(category)
    print("嵌入结果：", embed)