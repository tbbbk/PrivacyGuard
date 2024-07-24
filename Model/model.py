import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, degree
from torch.utils.data import DataLoader, Dataset, Subset
import random
import numpy as np
import yaml
import sys
import networkx as nx
from torch_geometric.nn.conv import GATConv, GCNConv

with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import Model.node2embed
import Model.relation2embed
import utils.load_data as ld
from tqdm import tqdm

class GCNConv1(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, link_embed=None, with_link = False):
        if(with_link == False):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            x = self.lin(x)
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return self.propagate(edge_index, x=x, norm=norm)
        elif(with_link == True):
            updated_x = x.clone()  # 创建一个副本用于更新
            for i in range(edge_index.size(1)):
                updated_x[edge_index[0][i]] += link_embed[i]
                updated_x[edge_index[1][i]] += link_embed[i]
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            updated_x = self.lin(updated_x)
            link_embed = self.lin(link_embed)
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return self.propagate(edge_index, x=updated_x, norm=norm, link_embed=link_embed), link_embed

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class GCN4line_graph(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(25, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv5 = GCNConv(64, 5)
        self.conv3 = GCNConv(51, 64)
        self.conv4 = GCNConv(64, 5)
        self.fc = torch.nn.Linear(10, 2)

    def forward(self, data):
        batch_orgin, batch_line = data[0], data[1]
        x_origin, edge_index_origin, ptr_origin = batch_orgin.x, batch_orgin.edge_index, batch_orgin.batch
        x_line, edge_index_line, ptr_line = batch_line.x, batch_line.edge_index.int(), batch_line.batch

        x_origin = self.conv1(x_origin, edge_index_origin)
        x_origin = F.relu(x_origin)
        x_origin = self.conv2(x_origin, edge_index_origin)
        x_origin = F.relu(x_origin)
        x_origin = self.conv5(x_origin, edge_index_origin)
        # 下面这一步将一个图中所有节点的值融合为了一个值
        x_origin = global_mean_pool(x_origin, ptr_origin)

        x_line = self.conv3(x_line, edge_index_line)
        x_line = F.relu(x_line)
        x_line = self.conv4(x_line, edge_index_line)
        # 下面这一步将一个图中所有节点的值融合为了一个值
        x_line = global_mean_pool(x_line, ptr_line)

        x = torch.concat((x_origin, x_line), dim=1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# GCN 模型
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



class EarlyStopping():
    def __init__(self,patience=7,verbose=False,delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
    def __call__(self,val_loss,model,path):
        print("val_loss={}".format(val_loss))
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
        elif score < self.best_score+self.delta:
            self.counter+=1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter>=self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss,model,path)
            self.counter = 0
    def save_checkpoint(self,val_loss,model,path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'model_checkpoint.pth')
        self.val_loss_min = val_loss


class privacy_dataset(Dataset):
    def __init__(self, name="mosaic", load_embed_mode="nn_embed", line_graph=False):
        super(privacy_dataset, self).__init__()
        self.name = name
        with open('./configs/model.yaml') as f:
            model_info = yaml.safe_load(f)
        seeds_num = model_info["model"]["seed"]
        # 设置随机种子
        torch.manual_seed(seeds_num)
        self.num_categories = model_info["model"]["node"]["num_categories"]  # 类别的数量
        self.embedding_dim = model_info["model"]["node"]["embedding_dim"]  # 嵌入维度
        self.CategoryEmbedding = Model.node2embed.CategoryEmbedding(self.num_categories, self.embedding_dim)
        self.Relationship_num_relations = model_info["model"]["relationship"]["num_relations"]  # 类别的数量
        self.Relationship_embedding_dim = model_info["model"]["relationship"]["embedding_dim"]  # 嵌入维度
        self.RelationEmbedding = Model.relation2embed.RelationEmbedding(self.Relationship_num_relations, self.Relationship_embedding_dim)
        self.line_data_list = []
        self.line_graph = line_graph
        if(name == "mosaic"):
            print(f"processing {name}")
            self.mosaic_info = ld.load_mosaic_info(file_name="random_balance_custom_data_info.json")
            self.mosaic_prediction = ld.load_mosaic_prediction(file_name="random_balance_custom_prediction.json")
            self.data_list  = []
            for indx, prediction in tqdm(enumerate(self.mosaic_prediction.keys())):
                # print(prediction)
                node_label = self.CategoryEmbedding(torch.tensor(self.mosaic_prediction[str(indx)]['bbox_labels']), to = load_embed_mode).type(torch.float)
                link_embed = self.RelationEmbedding(torch.tensor(self.mosaic_prediction[str(indx)]['rel_labels']),  to = load_embed_mode).type(torch.float)
                y = torch.tensor(self.mosaic_info["all_true_label"][indx], dtype=torch.long)
                edge_index = torch.tensor(self.mosaic_prediction[str(indx)]['rel_pairs'], dtype=torch.long)
                # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
                data_obj = Data(x=node_label, link_embed=link_embed, y=y, edge_index=edge_index)

                self.data_list.append(data_obj)
        elif(name == "privacy_dataset_1000"):
            print(f"processing {name}")
            self.privacy_info = ld.load_privacy_info()
            self.privacy_prediction = ld.load_privacy_prediction()
            self.public_info = ld.load_public_info()
            self.public_prediction = ld.load_public_prediction()
            self.data_list  = []
            for indx, prediction in tqdm(enumerate(self.privacy_prediction.keys()),desc="creating dataset"):
                # print(prediction)
                node_label = self.CategoryEmbedding(torch.tensor(self.privacy_prediction[str(indx)]['bbox_labels']), to = load_embed_mode).type(torch.float)
                link_embed = self.RelationEmbedding(torch.tensor(self.privacy_prediction[str(indx)]['rel_labels']),  to = load_embed_mode).type(torch.float)
                y = torch.tensor(self.privacy_info["all_true_label"][indx], dtype=torch.long)
                edge_index = torch.tensor(self.privacy_prediction[str(indx)]['rel_pairs'], dtype=torch.long)
                # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
                data_obj = Data(x=node_label, link_embed=link_embed, y=y, edge_index=edge_index)
                self.data_list.append(data_obj)
            for indx, prediction in tqdm(enumerate(self.public_prediction.keys())):
                # print(prediction)
                node_label = self.CategoryEmbedding(torch.tensor(self.public_prediction[str(indx)]['bbox_labels']), to = load_embed_mode).type(torch.float)
                link_embed = self.RelationEmbedding(torch.tensor(self.public_prediction[str(indx)]['rel_labels']),  to = load_embed_mode).type(torch.float)
                y = torch.tensor(self.public_info["all_true_label"][indx], dtype=torch.long)
                edge_index = torch.tensor(self.public_prediction[str(indx)]['rel_pairs'], dtype=torch.long)
                # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
                data_obj = Data(x=node_label, link_embed=link_embed, y=y, edge_index=edge_index)
                self.data_list.append(data_obj)
        elif(name == "privacy_dataset_1000_privacy"):
            print(f"processing {name}")
            self.privacy_info = ld.load_privacy_info(file_name="random_balance_custom_data_info.json")
            self.privacy_prediction = ld.load_privacy_prediction(file_name="random_balance_custom_prediction.json")
            self.data_list  = []
            for indx, prediction in tqdm(enumerate(self.privacy_prediction.keys()), desc="creating dataset"):
                # print(prediction)
                node_label = self.CategoryEmbedding(torch.tensor(self.privacy_prediction[str(indx)]['bbox_labels']), to = load_embed_mode).type(torch.float)
                link_embed = self.RelationEmbedding(torch.tensor(self.privacy_prediction[str(indx)]['rel_labels']),  to = load_embed_mode).type(torch.float)
                y = torch.tensor(self.privacy_info["all_true_label"][indx], dtype=torch.long)
                edge_index = torch.tensor(self.privacy_prediction[str(indx)]['rel_pairs'], dtype=torch.long)
                # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
                data_obj = Data(x=node_label, link_embed=link_embed, y=y, edge_index=edge_index)
                self.data_list.append(data_obj)
        elif(name == "privacy_dataset_1000_public"):
            print(f"processing {name}")
            self.public_info = ld.load_public_info()
            self.public_prediction = ld.load_public_prediction()
            self.data_list  = []
            for indx, prediction in tqdm(enumerate(self.public_prediction.keys()), desc="creating dataset"):
                # print(prediction)
                node_label = self.CategoryEmbedding(torch.tensor(self.public_prediction[str(indx)]['bbox_labels']), to = load_embed_mode).type(torch.float)
                link_embed = self.RelationEmbedding(torch.tensor(self.public_prediction[str(indx)]['rel_labels']),  to = load_embed_mode).type(torch.float)
                y = torch.tensor(self.public_info["all_true_label"][indx], dtype=torch.long)
                edge_index = torch.tensor(self.public_prediction[str(indx)]['rel_pairs'], dtype=torch.long)
                # x 节点特征，linkembed 连接关系特征， y 是否隐私， edge index 临界表[[0, 1, 2], [1, 2, 3]]
                data_obj = Data(x=node_label, link_embed=link_embed, y=y, edge_index=edge_index)
                self.data_list.append(data_obj)
        else:
            print("error")
        
        if line_graph:
            data_list = self.data_list
            for index, data in enumerate(data_list):
                # 先声明一个原图
                G = nx.graph()
                for i in range(len(data.edge_index.shape[1])):
                    # 添加边，起始节点和关系嵌入
                    G.add_edges_from([(data.edge_index[0][i].item(), data.edge_index[1][i].item(), {'relation': data.link_embed[i]})])
                # 根据原图创建线图
                H = nx.line_graph(G)
                #  这段的解释见https://networkx.org/documentation/stable/reference/generated/networkx.generators.line.line_graph.html最底部
                H.add_nodes_from((node, G.edges[node]) for node in H)
                x = torch.tensor([list(node[1]['relation']) for node in H.nodes(data=True)], dtype=torch.float32)
                edge_start = []
                edge_end = []
                node_mapping = {node: idx for idx, node in enumerate(H.nodes)}
                for edge in H.edges:
                    edge_start.append(node_mapping[edge[0]])
                    edge_end.append(node_mapping[edge[1]])
                edge_index = torch.tensor([edge_start, edge_end])
                data_line = Data(x=x, edge_index=edge_index, y=data.y, edge_weight=None)
                self.line_data_list.append(data_line)
            
                
            
            
            
    def __getitem__(self, index):
        if self.line_graph:
            return self.data_list[index], self.line_data_list[index]
        batch = self.data_list[index]
        # 返回输入和输出数据
        return batch

    def __len__(self):
        return len(self.data_list)

    def get_dataset(self):
        if self.line_graph:
            return [self.data_list, self.line_data_list]
        return self.data_list
    


if __name__ == "__main__":
    # 示例数据
    with open('./configs/model.yaml') as f:
        model_info = yaml.safe_load(f)
    seeds_num = model_info["model"]["seed"] 
    # 设置随机种子
    torch.manual_seed(seeds_num)
    np.random.seed(seeds_num)
    random.seed(seeds_num)


    # 创建完整数据集
    full_dataset = privacy_dataset("mosaic")

    # 计算数据集大小
    dataset_size = len(full_dataset)

    # 计算划分索引
    train_size = int(0.8 * dataset_size)  # 80% 作为训练集
    test_size = dataset_size - train_size  # 剩余部分作为测试集

    # 创建索引并打乱顺序
    indices = torch.randperm(dataset_size)
    random.shuffle(indices)

    # 划分训练集和测试集
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 创建训练集和测试集的子集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
