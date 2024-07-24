import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv, GCNConv
from torch_geometric.loader import DataLoader,NeighborLoader
from tqdm import tqdm
import random
import numpy as np
import yaml
import sys
from torch.utils.data import random_split
with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import Model.node2embed
import Model.relation2embed
import utils.load_data as ld
import time
from utils.draw_confusion_matrix import generate_confusion_matrix
from utils.calculate_index import calculate_index, ROC_PR

import os
w1 = 1 # 预测为1时的权重
w0 = 1.1  # 预测为0时的权重

def load_data(load_embed_mode="nn_embed", dataset_name = "mosaic", is_balance = True, is_random = True):
    print(f"datasetname:{dataset_name}")
    print(f"is_balance = {is_balance}, is_random = {is_random}")
    print(f"embed_mode={load_embed_mode}")
    if(is_balance == True):
        if(is_random == False):
            info_name = "reltr_custom_data_info.json"
            pred_name = "reltr_custom_prediction.json"
        elif(is_random == True):
            info_name = "reltr_custom_data_info.json"
            pred_name = "reltr_custom_prediction.json"
    elif(is_balance == False):
        info_name = "reltr_custom_data_info.json"
        pred_name = "reltr_custom_prediction.json"
    if(dataset_name == "mosaic"):
        mosaic_info = ld.load_mosaic_info("all", file_name=info_name)
        mosaic_pred = ld.load_mosaic_prediction("all", file_name = pred_name)
    elif(dataset_name == "privacy_1000"):
        mosaic_info = ld.load_privacy_info("all", file_name=info_name)
        mosaic_pred = ld.load_privacy_prediction("all", file_name=pred_name)
    with open('./configs/model.yaml') as f:
            model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"]
    # 设置随机种子
    torch.manual_seed(seeds_num)
    num_categories = model_info["model"]["node"]["num_categories"]  # 类别的数量
    embedding_dim = model_info["model"]["node"]["embedding_dim"]  # 嵌入维度
    CategoryEmbedding = Model.node2embed.CategoryEmbedding(num_categories, embedding_dim)
    Relationship_num_relations = model_info["model"]["relationship"]["num_relations"]  # 类别的数量
    Relationship_embedding_dim = model_info["model"]["relationship"]["embedding_dim"]  # 嵌入维度
    RelationEmbedding = Model.relation2embed.RelationEmbedding(Relationship_num_relations, Relationship_embedding_dim)
    data_list = []
    for index, prediction in enumerate(mosaic_pred):
        node_label = CategoryEmbedding(torch.tensor(mosaic_pred[str(index)]['bbox_labels']), to = load_embed_mode).type(torch.float)
        link_embed = RelationEmbedding(torch.tensor(mosaic_pred[str(index)]['rel_labels']),  to = load_embed_mode).type(torch.float)
        y = torch.tensor(mosaic_info["all_true_label"][index], dtype=torch.long)
        edge_index = torch.tensor(mosaic_pred[str(index)]['rel_pairs'], dtype=torch.long)
        data = HeteroData()
        data['object'].x = node_label
        data['relation'].x = link_embed
        data['object'].y = y
        train_size = int(0.8 * len(node_label))
        data['object'].train_mask=torch.tensor([i for i in range(0, train_size)])
        data['object'].val_mask=torch.tensor([i for i in range(train_size, len(node_label))])
        # 添加边类型和边索引
        node_edge_index0 = edge_index[0].tolist()
        node_edge_index1 = edge_index[1].tolist()
        link_edge_index =  [i for i in range(0, len(link_embed))]
        data['object','to', 'relation'].edge_index = torch.tensor([node_edge_index0,
                                                    link_edge_index], dtype=torch.long) # @TODO
        data['relation','to', 'object'].edge_index = torch.tensor([link_edge_index,
                                                    node_edge_index1], dtype=torch.long)
        data_list.append(data)
    return data_list

class HAN(nn.Module):
    def __init__(self, in_channels, hidden_channels,linear_layer, linear_hidden_layer, linear_hidden_layer2, out_channels, metadata):
        super(HAN, self).__init__()
        self.conv1 = HANConv(in_channels, hidden_channels, metadata, heads=2)
        self.conv2 = HANConv(hidden_channels, linear_layer, metadata, heads=2)
        self.lin = nn.Linear(linear_layer, linear_hidden_layer)
        self.lin2 = nn.Linear(linear_hidden_layer, linear_hidden_layer2)
        self.lin3 = nn.Linear(linear_hidden_layer2, out_channels)
    
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
            # print(x_dict, '\n===\n', edge_index_dict)
            # input()
            x = self.conv1(x_dict, edge_index_dict)
            x_res = x['object']
            x = self.conv2(x, edge_index_dict)
            x['object'] += x_res
            x = x['object'] 
            x = self.lin(x)
            x = self.lin2(x)
            x = self.lin3(x)
            x = torch.sigmoid(x)
        return x
    


def test(model, is_res=False, is_attention=False, is_L2loss=False):
    model.eval()
    with torch.no_grad():
        correct = 0
        test_num = 0
        pred_node, y_node, ans_origin_value = [], [], []
        for i, graph in enumerate(dataloader):
            graph = graph.to(device)
            val_mask = graph['object'].val_mask
            y = graph['object'].y.to(device)
            out = model(graph, is_res = is_res, is_attention = is_attention)
            ans = torch.tensor([1 if x > 0.5 else 0 for x in out[val_mask]]).to(device)
            ans_origin = torch.tensor([x for x in out[val_mask]]).to(device)
            ans_origin_value.append(ans_origin)
            pred_node.append(ans.tolist())
            y_node.append(y[val_mask].tolist())
            correct = correct + int(ans.eq(y[val_mask]).sum().item())
            test_num += len(val_mask)
        acc = correct / test_num
        
    return acc, pred_node, y_node, ans_origin_value


if __name__ == "__main__":
    with open('./configs/model.yaml') as f:
            model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"]
    # 设置随机种子
    torch.manual_seed(seeds_num)
    datalist = load_data(dataset_name="privacy_1000",is_balance = True, is_random = True)
    dataloader = DataLoader(dataset=datalist, batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = (['object', 'relation'], [('object', 'to', 'relation'), ('relation', 'to', 'object')])
    model = HAN(-1, 32, 32, 64, 64, 1, metadata=metadata).to(device)
    model = torch.load(f'/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-13-19-06/195.pth')   # @TODO 泛化最佳
    acc, pred, y, pred_origin = test(model, is_res = True, is_attention = False, is_L2loss = True)
    pred = torch.cat([torch.tensor(sublist) for sublist in pred], dim = 0)
    y = torch.cat([torch.tensor(sublist) for sublist in y], dim = 0)
    pred_origin = torch.cat([torch.tensor(sublist) for sublist in pred_origin], dim = 0)
    calculate_index(pred, y)