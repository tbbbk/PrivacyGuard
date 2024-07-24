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
import cv2
w1 = 1 # 预测为1时的权重
w0 = 1.1  # 预测为0时的权重

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
    # 设置随机种子
    torch.manual_seed(seeds_num)
    num_categories = model_info["model"]["node"]["num_categories"]  # 类别的数量
    embedding_dim = model_info["model"]["node"]["embedding_dim"]  # 嵌入维度
    CategoryEmbedding = Model.node2embed.CategoryEmbedding(num_categories, embedding_dim)
    Relationship_num_relations = model_info["model"]["relationship"]["num_relations"]  # 类别的数量
    Relationship_embedding_dim = model_info["model"]["relationship"]["embedding_dim"]  # 嵌入维度
    RelationEmbedding = Model.relation2embed.RelationEmbedding(Relationship_num_relations, Relationship_embedding_dim)
    data_list = []
    img_dir = []
    img_info = []
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
        data_list.append(data)
        img_dir.append(mosaic_info["idx_to_files"][index])
        img_info.append(mosaic_pred[str(index)]["bbox"])
    return data_list, img_dir, img_info

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


    
def count_files_in_folder(folder_path):
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

def process_box_list(box_list):
    new_box_list = []
    
    # 定义函数用于判断两个框是否相交
    def is_intersecting(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min)
    
    # 定义函数用于判断一个框是否被另一个框包含
    def is_contained(box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        return x1_min >= x2_min and y1_min >= y2_min and x1_max <= x2_max and y1_max <= y2_max
    
    # 遍历框列表
    for box in box_list:
        should_add = True
        for index, existing_box in enumerate(new_box_list):
            if is_contained(box, existing_box):
                # 如果当前框被已存在的框包含，则不添加当前框
                should_add = False
                break
            elif is_intersecting(box, existing_box):
                # 如果当前框与已存在的框相交，则合并为一个框
                x_min = min(box[0], existing_box[0])
                y_min = min(box[1], existing_box[1])
                x_max = max(box[2], existing_box[2])
                y_max = max(box[3], existing_box[3])
                new_box_list[index] = [x_min, y_min, x_max, y_max]
                should_add = False
                break
        if should_add:
            new_box_list.append(box)
    
    return new_box_list



def test(model, model_path, img_dir, img_info, is_res=False, is_attention=False, is_L2loss=False, save_dir = "/home/zhuohang/SGG-GCN/han/visual"):
    metadata = (['object', 'relation'], [('object', 'to', 'relation'), ('relation', 'to', 'object')])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model from the provided path
    model = HAN(-1, 32, 32, 64, 64, 1, metadata=metadata).to(device)
    model = torch.load(model_path, map_location=device)

    # print(f"Test time: {training_time}")
    print("Model: HAN(-1, 32, 32, 64, 64, 1, metadata=metadata).to(device)")
    print(model)
    print(f"is_res={is_res}, is_attention={is_attention}, is_L2loss={is_L2loss}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    min_epochs = 5
    best_val_acc = 0
    final_best_acc = 0

    model.eval()
    with torch.no_grad():
        correct = 0
        test_num = 0
        for i, graph in enumerate(train_loader):
            graph = graph.to(device)
            # print(graph)
            train_mask, val_mask, test_mask = graph['object'].train_mask, graph['object'].val_mask, graph['object'].test_mask
            y = graph['object'].y.to(device)
            out = model(graph, is_res=is_res, is_attention=is_attention)
            # print(f"true: {y}")
            # print(f"pred: {out.squeeze()}")
            dir = img_dir[i]
            bbox = img_info[i]           
            ans = torch.tensor([1 if x > 0.5 else 0 for x in out[test_mask]]).to(device)
            # print(ans)
            # 读取照片
            image = cv2.imread(dir)
            box_list = []
            for j in range(len(ans)):
                if(ans[j] == 1):
                    box_list.append(bbox[j])
            box_list = process_box_list(box_list)
            for k in range(len(box_list)):
                box = box_list[k]
                x1, y1, x2, y2 = box
                # 确定边界框的整数坐标
                w,h,c = image.shape
                # 确定边界框的整数坐标
                x1, y1, x2, y2 =  min(int(x1), h) , min(int(y1),w), min(int(x2), h) , min(int(y2),w)
                if(x1 < x2 and y1 < y2):
                    bbox_img = image[y1:y2, x1:x2]
                    
                    # 绘制边界框
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 在图像上绘制矩形
                        
            name = count_files_in_folder(save_dir)
            cv2.imwrite(f"{save_dir}/{name}.jpg", image)




if __name__ == "__main__":
    datalist, img_dir, img_info = load_data(dataset_name="privacy_1000",is_balance = True, is_random = True)
    print(len(datalist))
    print(datalist[0])
    train_loader = DataLoader(dataset=datalist, batch_size=1, shuffle=False)
    model_path = "/home/zhuohang/SGG-GCN/pth/hanpth/hanpth_nobalance/2024-03-05-13-19-06/115.pth"
    test(train_loader,model_path,img_dir, img_info, is_res = True, is_attention = False, is_L2loss = True, save_dir="/home/zhuohang/SGG-GCN/han/visual")

