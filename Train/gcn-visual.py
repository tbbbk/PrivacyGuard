import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Subset
import sys
import yaml
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os
import cv2
with open("./configs/model.yaml", "r") as f:
    model_yaml = yaml.safe_load(f)

project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)

import utils.load_data as ld
from Model.model import GCN, privacy_dataset, EarlyStopping, GAT

def count_files_in_folder(folder_path):
    file_count = 0
    for _, _, files in os.walk(folder_path):
        file_count += len(files)
    return file_count

# 创建数据加载器
def create_data_loader(data_list, batch_size, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

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
# 测试函数
def test(model, data_loader, device, img_dir, save_dir, dataset_pred):
    pred_node, y_node, ans_origin_value = [], [], []
    model.eval()
    total_len = 0
    correct = 0
    for idx,batch in (enumerate(data_loader)):
        batch = batch.to(device)
        output = model(batch)
        pred = output.argmax(dim=1).tolist()
        # true_label = batch.y.tolist()
        bbox = dataset_pred[str(idx)]["bbox"]
        image_dir = img_dir[idx]
        image = cv2.imread(image_dir)
        box_list = []
        for j in range(len(pred)):
            if(pred[j] == 1):
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
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="privacy_dataset_1000_privacy",help='the dataset want to change(privacy_dataset_1000, mosaic, privacy_dataset_1000_privacy, privacy_dataset_1000_public)')
    parser.add_argument('--embed_mode', default="nn_embed", help="nn_embed or onehot")
    parser.add_argument('--top_n', default=30, type=int , help="int num(0-80)")
    parser.add_argument("--least_trust", default=0.5,type=float, help="0-1")
    parser.add_argument("--relation_node", default="add", help="add or multi")
    parser.add_argument("--epoch_num", default=200, type=int , help="int epoch num")
    parser.add_argument("--train_ratio", default=0.8, type= float, help="0-1")
    parser.add_argument("--lr", default=0.01, type=float, help="0-1")
    parser.add_argument("--save_num", default=10, type=int, help="How many intervals to save(1-inf)")
    parser.add_argument("--batch_size", default=100, type=int, help="batch_size")

    training_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_folder_path = f"/home/zhuohang/SGG-GCN/pth/gcn_pth/{training_time}"

    args = parser.parse_args()
    dataset_name = args.dataset
    embed_mode = args.embed_mode
    top_n = args.top_n
    least_trust = args.least_trust
    relation_node = args.relation_node
    epoch_num = args.epoch_num
    model_train_ratio = args.train_ratio
    model_lr = args.lr
    save_num = args.save_num
    model_batch_size= args.batch_size
    # 示例数据
    with open('./configs/model.yaml') as file:
        model_info = yaml.safe_load(file)

    seeds_num = model_info["model"]["seed"]
    torch.manual_seed(seeds_num)

    # 创建GCN模型
    node_features_dim = model_info["model"]["node"]["embedding_dim"]
    node_num_categories = model_info["model"]["node"]["num_categories"]

    edge_embed_dim = model_info["model"]["relationship"]["embedding_dim"]
    edge_num_relations = model_info["model"]["relationship"]["num_relations"]

    input_layer_dim = model_info["model"]["input_layer"]["input_layer_dim"]
    hidden_layer_dim = model_info["model"]["hidden_layer"]["hidden_layer_dim"]
    output_layer_dim = model_info["model"]["output_layer"]["output_dim"]
    with_link = model_info["model"]["with_link"]
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_info = ld.load_privacy_info(file_name="balance_custom_data_info.json")
    dataset_pred = ld.load_privacy_prediction(file_name="balance_custom_prediction.json")
    
    img_dir = dataset_info["idx_to_files"]
    # 创建完整数据集
    full_dataset = privacy_dataset(dataset_name)
    # 创建数据加载器
    data_loader = create_data_loader(full_dataset, batch_size=model_batch_size, shuffle=True)
    save_dir = "/home/zhuohang/SGG-GCN/Train/GCN-Visual"
    # 创建模型
    model = GCN(in_channels=input_layer_dim, hidden_channels=hidden_layer_dim, out_channels=output_layer_dim).to(device)
    model.load_state_dict(torch.load('/home/zhuohang/SGG-GCN/pth/gcn_pth/2024-03-05-12-40-32/GCN_model_200.pth'))
    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)

    
    test(model, full_dataset, device, img_dir, save_dir, dataset_pred)

    