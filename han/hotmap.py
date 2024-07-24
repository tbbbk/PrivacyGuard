import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv
from torch_geometric.loader import DataLoader,NeighborLoader
from tqdm import tqdm
import random
import numpy as np
import yaml
import sys
with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import Model.node2embed
import Model.relation2embed
import utils.load_data as ld
import time
import os
w1 = 1 # 预测为1时的权重
w0 = 1.1  # 预测为0时的权重
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
import shutil
def load_data(load_embed_mode="nn_embed", dataset_name = "mosaic", is_balance = True, is_random = True):
    print(f"datasetname:{dataset_name}")
    print(f"is_balance = {is_balance}, is_random = {is_random}")
    print(f"embed_mode={load_embed_mode}")
    if(is_balance == True):
        if(is_random == False):
            info_name = "balance_custom_data_info.json"
            pred_name = "balance_custom_prediction.json"
        elif(is_random == True):
            info_name = "random_balance_custom_data_info.json"
            pred_name = "random_balance_custom_prediction.json"
    elif(is_balance == False):
        info_name = "custom_data_info.json"
        pred_name = "custom_prediction.json"
    if(dataset_name == "mosaic"):
        mosaic_info = ld.load_mosaic_info("all", file_name=info_name)
        mosaic_pred = ld.load_mosaic_prediction("all", file_name = pred_name)
    elif(dataset_name == "privacy_1000"):
        mosaic_info = ld.load_privacy_info("all", file_name=info_name)
        mosaic_pred = ld.load_privacy_prediction("all", file_name=pred_name)
    with open('./configs/model.yaml') as f:
            model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"]
    cat_info = mosaic_info["ind_to_classes"]
    img_dir = mosaic_info["idx_to_files"]
    # 设置随机种子
    torch.manual_seed(seeds_num)
    num_categories = model_info["model"]["node"]["num_categories"]  # 类别的数量
    embedding_dim = model_info["model"]["node"]["embedding_dim"]  # 嵌入维度
    CategoryEmbedding = Model.node2embed.CategoryEmbedding(num_categories, embedding_dim)
    Relationship_num_relations = model_info["model"]["relationship"]["num_relations"]  # 类别的数量
    Relationship_embedding_dim = model_info["model"]["relationship"]["embedding_dim"]  # 嵌入维度
    RelationEmbedding = Model.relation2embed.RelationEmbedding(Relationship_num_relations, Relationship_embedding_dim)
    data_list = []
    cat_label = []
    for index, prediction in enumerate(mosaic_pred):
        node_label = CategoryEmbedding(torch.tensor(mosaic_pred[str(index)]['bbox_labels']), to = load_embed_mode).type(torch.float)
        link_embed = RelationEmbedding(torch.tensor(mosaic_pred[str(index)]['rel_labels']),  to = load_embed_mode).type(torch.float)
        y = torch.tensor(mosaic_info["all_true_label"][index], dtype=torch.long)
        cat_label.append(mosaic_pred[str(index)]['bbox_labels'])
        edge_index = torch.tensor(mosaic_pred[str(index)]['rel_pairs'], dtype=torch.long)
        # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
        data = HeteroData()
        data['object'].x = node_label
        data['relation'].x = link_embed
        data['object'].y = y

        data['object'].train_mask=torch.tensor([i for i in range(0, len(node_label))])
        data['object'].val_mask=torch.tensor([i for i in range(0, len(node_label))])
        data['object'].test_mask=torch.tensor([i for i in range(0, len(node_label))])
        # 添加边类型和边索引
        node_edge_index0 = edge_index[0].tolist()
        node_edge_index1 = edge_index[1].tolist()
        link_edge_index =  [i for i in range(0, len(link_embed))]
        # torch.arange(0, len(link_embed),dtype=torch.long)
        data['object','to', 'relation'].edge_index = torch.tensor([node_edge_index0,
                                                    link_edge_index], dtype=torch.long) 
        data['relation','to', 'object'].edge_index = torch.tensor([link_edge_index,
                                                    node_edge_index1], dtype=torch.long)
        data_list.append(data)
    return data_list, cat_label, cat_info, img_dir

class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels,linear_layer, linear_hidden_layer, linear_hidden_layer2, out_channels, metadata):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads=2)
        self.conv2 = HANConv(hidden_channels, linear_layer, metadata, heads=2)
        self.lin = nn.Linear(linear_layer, linear_hidden_layer)
        self.lin2 = nn.Linear(linear_hidden_layer, linear_hidden_layer2)
        self.lin3 = nn.Linear(linear_hidden_layer2, out_channels)
    # 基础 
    def forward(self, data, is_res=False, is_attention=False):
        if(is_res == False and is_attention == False):
            x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
            x = self.conv1(x_dict, edge_index_dict)
            x = self.conv2(x, edge_index_dict)
            x = self.lin(x['object'])
            x = self.lin2(x)
            x = self.lin3(x)
            x = torch.sigmoid(x)
        elif(is_res == True and is_attention == False):
            x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
            x = self.conv1(x_dict, edge_index_dict)
            x_res = x['object']
            x = self.conv2(x, edge_index_dict)
            x['object'] += x_res
            x = x['object']  # Assuming 'object' is the key you want to apply attention to
            # x = x.permute(1, 0, 2)  # Reshape input for attention layer
            # x, _ = self.attention(x, x, x)  # Apply self-attention
            # x = x.permute(1, 0, 2)  # Reshape back to original shape
            x = self.lin(x)
            x = self.lin2(x)
            x = self.lin3(x)
            x = torch.sigmoid(x)
        return x

# 初始化模型
metadata = (['object', 'relation'], [('object', 'to', 'relation'), ('relation', 'to', 'object')])
num_classes = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HAN(-1, 32, 32, 64, 64, 1, metadata=metadata).to(device)
model_path = "/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-13-19-06/115.pth"
model = torch.load(model_path, map_location=device)


# 定义一个字典来存储每一层的输出
layer_outputs = {}

# 定义一个钩子函数
def get_layer_output(module, input, output):
    layer_outputs[str(module)] = output

# 注册钩子
for layer in model.children():
    layer.register_forward_hook(get_layer_output)


datalist,my_cat_label,my_cat_info, img_dir = load_data(dataset_name="privacy_1000",is_balance = True, is_random = False)
# print(len(datalist))
# print(datalist[0])
train_loader = DataLoader(dataset=datalist, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# dataset = data
graph = datalist[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
min_epochs = 5
best_val_acc = 0
final_best_acc = 0

is_res = True
is_attention = False
is_L2loss = True

# 计算模型输出（在这个过程中，钩子函数将会被调用）
model.eval()
with torch.no_grad():
    for i, graph in tqdm(enumerate(train_loader)):
        now_cat_label = [my_cat_info[j] for j in my_cat_label[i]]
        graph = graph.to(device)
        train_mask, val_mask, test_mask = graph['object'].train_mask, graph['object'].val_mask, graph['object'].test_mask
        label = graph['object']
        y = graph['object'].y.to(device)
        out = model(graph, is_res=is_res, is_attention=is_attention)
        if not os.path.exists(f"/home/zhuohang/SGG-GCN/draw/img/{i}"):
            os.makedirs(f"/home/zhuohang/SGG-GCN/draw/img/{i}")
        # 绘制每一层的输出热力图，每个热力图包含所有节点的输出
        idx = 0
        idx_img_dir = img_dir[i]
        # shutil.copyfile(idx_img_dir, f"/home/zhuohang/SGG-GCN/draw/img/{i}/image.jpg")
        for layer, output_dict in layer_outputs.items():
            # for node_type, output in output_dict.items():
                # print(output_dict.keys())
                if(idx == 0):
                    output = output_dict["object"].cpu()
                else:
                    output = output_dict.cpu()
                idx += 1
                output = output.detach().numpy()  # 将输出转换为numpy数组
                output= np.mean(output, axis=1, keepdims=True)
                output = output[:10]
                now_cat_label = now_cat_label[:10]
                #output= np.sum(output, axis=1, keepdims=True)
                #output= np.max(output, axis=1, keepdims=True)
                #output= np.min(output, axis=1, keepdims=True)

                # output= np.median(output, axis=1)[:, np.newaxis]
                plt.figure(figsize=(10, 8))
                # 直接绘制热力图，无需reshape。每行代表一个节点的输出特征。
                ax = sns.heatmap(output, annot=True, fmt="f", cmap='viridis',  yticklabels=now_cat_label)
                plt.title(f"Heatmap for {layer}")
                
                # 构建图层名字的字符串表示，用于文件名
                layer_name = str(layer).replace('\n', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
                
                # 保存热力图到指定路径
                img_path = f"/home/zhuohang/SGG-GCN/draw/img/{i}/{layer_name}.jpg"
                plt.savefig(img_path)
                plt.close()  # 关闭图表以释放内存




