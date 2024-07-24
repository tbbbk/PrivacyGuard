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
            info_name = "random_balance_custom_data_info.json"
            pred_name = "zoom_random_balance_custom_prediction.json"
        elif(is_random == True):
            info_name = "random_balance_custom_data_info.json"
            pred_name = "zoom_random_balance_custom_prediction.json"
    elif(is_balance == False):
        info_name = "random_balance_custom_data_info.json"
        pred_name = "zoom_random_balance_custom_prediction.json"
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
        # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
        data = HeteroData()
        data['object'].x = node_label
        data['relation'].x = link_embed
        data['object'].y = y
        train_size = int(0.2 * len(node_label))
        # data['object'].train_mask=torch.tensor([i for i in range(0, train_size)])
        # data['object'].val_mask=torch.tensor([i for i in range(train_size, len(node_label))])
        data['object'].train_mask=torch.tensor([i for i in range(0, len(node_label))])
        data['object'].val_mask=torch.tensor([i for i in range(0, len(node_label))])
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
    return data_list

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
    

def weighted_binary_cross_entropy(pred, y, n, w1, w0, model=None, lambda_reg=None, is_L2loss=False):
    if(is_L2loss == False):
        pred = torch.clamp(pred, 0.0001, 0.9999)  # 限制预测值的范围
        loss = -w1 * y * torch.log(pred) - w0 * (1 - y) * torch.log(1 - pred)
        loss = torch.mean(loss)
    elif(is_L2loss == True):
        pred = torch.clamp(pred, 0.0001, 0.9999)  # 限制预测值的范围
        loss = -w1 * y * torch.log(pred) - w0 * (1 - y) * torch.log(1 - pred)
        loss = torch.mean(loss)

        # 添加L2正则化项
        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += torch.norm(param, p=2)
        loss += lambda_reg * regularization_loss
    return loss



def train(train_loader, is_res=False, is_attention=False, is_L2loss=False):
    training_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_folder_path = f"/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/{training_time}/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    num_classes = 2
    print(f"train time : {training_time}")
    print(model)
    print(f"is_res={is_res}, is_attention={is_attention}, is_L2loss={is_L2loss}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
 
    model.train()
    for epoch in tqdm(range(20)):
        for graph in tqdm(train_loader):
            graph = graph.to(device)
            train_mask = graph['object'].train_mask
            y = graph['object'].y.to(device)
            f = model(graph, is_res = is_res, is_attention = is_attention)
            loss = weighted_binary_cross_entropy(f[train_mask].squeeze(), y[train_mask], len(train_mask), w1, w0, model, 0.1, is_L2loss = is_L2loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        val_acc, _, _, _ = test(model, is_res = is_res, is_attention = is_attention, is_L2loss = is_L2loss)
        if epoch % 5 == 0:
            # 保存整个模型
            torch.save(model, save_folder_path+f'{epoch}.pth')
        elif epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = val_acc
            torch.save(model, save_folder_path+f'best.pth')
        tqdm.write('Epoch{} train_loss {} val_acc {}'
                   .format(epoch, loss.item(), val_acc))


def test(model, is_res=False, is_attention=False, is_L2loss=False):
    model.eval()
    with torch.no_grad():
        correct = 0
        test_num = 0
        pred_node, y_node, ans_origin_value = [], [], []
        for i, graph in enumerate(val_loader):
            graph = graph.to(device)
            val_mask = graph['object'].val_mask
            y = graph['object'].y.to(device)
            out = model(graph, is_res = is_res, is_attention = is_attention)
            loss = weighted_binary_cross_entropy(out[val_mask].squeeze(), y[val_mask], len(val_mask), w1, w0, model, 0.1, is_L2loss = is_L2loss)
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
    # datalist = load_data(dataset_name="mosaic",is_balance = True, is_random = True)
    dataset_size = len(datalist)
    # 计算训练集和验证集的大小
    train_size = int(0.5 * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(datalist, [train_size, val_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    # val_loader = DataLoader(dataset=datalist, batch_size=100, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata = (['object', 'relation'], [('object', 'to', 'relation'), ('relation', 'to', 'object')])
    model = HAN(-1, 32, 32, 64, 64, 1, metadata=metadata).to(device)
    # train(train_loader, is_res = True, is_attention = False, is_L2loss = True)
    # # for i in range(0, 200, 5):
    # # model = torch.load(f'/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-12-09-24/{i}.pth')
    # # model = torch.load(f'/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-13-19-06/195.pth')   # @TODO 泛化最佳
    model = torch.load(f'/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-13-19-06/115.pth') # @TODO privacy 最佳
    # # model = torch.load(f'/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-12-09-24/best.pth')
    acc, pred, y, pred_origin = test(model, is_res = True, is_attention = False, is_L2loss = True)
    pred = torch.cat([torch.tensor(sublist) for sublist in pred], dim = 0)
    y = torch.cat([torch.tensor(sublist) for sublist in y], dim = 0)
    pred_origin = torch.cat([torch.tensor(sublist) for sublist in pred_origin], dim = 0)
    # print(f'============第{i}epoch==============')
    calculate_index(pred, y)