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
w1 = 50  # 预测为1时的权重
w0 = 1  # 预测为0时的权重

def load_data(load_embed_mode="nn_embed", dataset_name = "mosaic"):
    if(dataset_name == "mosaic"):
        mosaic_info = ld.load_mosaic_info("all")
        mosaic_pred = ld.load_mosaic_prediction("all")
    elif(dataset_name == "privacy_1000"):
        mosaic_info = ld.load_privacy_info("all")
        mosaic_pred = ld.load_privacy_prediction("all")
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
        data['object', 'to', 'object'].edge_index = torch.tensor([node_edge_index0, node_edge_index1])
        # data['relation', 'to', 'relation'].edge_index = torch.tensor([])
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
    def forward(self, data):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict
        x = self.conv1(x_dict, edge_index_dict)
        x = self.conv2(x, edge_index_dict)
        x = self.lin(x['object'])
        x = self.lin2(x)
        x = self.lin3(x)
        x = torch.sigmoid(x)
        # _,x = x.max(dim=1)
        return x


def weighted_binary_cross_entropy(pred, y, n, w1, w0):
    pred = torch.clamp(pred, 0.0001, 0.9999)  # 限制预测值的范围
    loss = -w1 * y * torch.log(pred) - w0 * (1 - y) * torch.log(1 - pred)
    loss = torch.mean(loss)
    return loss

def train(train_loader):
    # model_path = './hanpth/nobalance/link'
    training_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_folder_path = f"/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_withlink/{training_time}/"
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    metadata = (['object', 'relation'], [('object', 'to', 'relation'), ('relation', 'to', 'object'),('object','to', 'object'), ('relation', 'to', 'relation')])
    num_classes = 2
    # model = HAN(-1, 64, 32, 16, 8, 1, metadata=metadata).to(device)
    model = HAN(-1, 128, 64, 128, 64, 1, metadata=metadata).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0
 
    model.train()
    for epoch in tqdm(range(100)):
        for graph in tqdm(train_loader):
            graph = graph.to(device)
            train_mask, val_mask, test_mask = graph['object'].train_mask, graph['object'].val_mask, graph['object'].test_mask
            y = graph['object'].y.to(device)
            f = model(graph)
            loss = weighted_binary_cross_entropy(f[train_mask].squeeze(), y[train_mask], len(train_mask), w1, w0)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        val_acc, val_loss = test(model, val_mask)
        test_acc, test_loss = test(model, test_mask)
        if epoch % 5 == 0:
            # 保存整个模型
            torch.save(model, save_folder_path+f'{epoch}.pth')

        if epoch + 1 > min_epochs and val_acc > best_val_acc:
            best_val_acc = val_acc
            final_best_acc = test_acc
        tqdm.write('Epoch{:3d} train_loss {:.5f} val_acc {:.3f} test_acc {:.3f}'
                   .format(epoch, loss.item(), val_acc, test_acc))

    return final_best_acc


def test(model, mask):
    model.eval()
    with torch.no_grad():
        correct = 0
        test_num = 0
        for i, graph in enumerate(train_loader):
            graph = graph.to(device)
            train_mask, val_mask, test_mask = graph['object'].train_mask, graph['object'].val_mask, graph['object'].test_mask
            y = graph['object'].y.to(device)
            out = model(graph)
            loss = weighted_binary_cross_entropy(out[mask].squeeze(), y[mask], len(mask), w1, w0)
            loss.requires_grad = True
            if i % 100 == 0:
                print(f"true:{y}")
                print(f"pred:{out.squeeze()}")
            ans = torch.tensor([1 if x > 0.8 else 0 for x in out[mask]]).to(device)

            correct = correct + int(ans.eq(y[mask]).sum().item())
            test_num += len(mask)
        acc = correct / test_num

    return acc, loss.item()


if __name__ == "__main__":
    datalist = load_data()
    print(len(datalist))
    print(datalist[0])
    train_loader = DataLoader(dataset=datalist, batch_size=100, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = data
    graph = datalist[0]
    final_best_acc = train(train_loader)
    print('HAN Accuracy:', final_best_acc)