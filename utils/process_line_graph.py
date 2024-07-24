import math
import matplotlib.pyplot as plt
import sys
import yaml
with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import utils.load_data as ld
import cv2
import json
import argparse
from tqdm import tqdm
import random

input_pred_dir = '/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/custom_prediction.json'
input_info_dir = '/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/custom_data_info.json'
output_pred_dir = '/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/line_custom_prediction.json'
output_info_dir = '/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/line_custom_data_info.json'
with open(input_info_dir, "r") as f:
    info = json.load(f)
with open(input_pred_dir, "r") as f:
    pred = json.load(f)

all_true_label = info['all_true_label']
count = len(all_true_label)
node_relation_node = []
node_relation_node_true_label = []
# count2 = len(pred)
for i in tqdm(range(count)):
    pred_i = pred[str(i)]
    true_label = all_true_label[i]
    rel_pairs0 = pred_i["rel_pairs"][0]
    rel_pairs1 = pred_i["rel_pairs"][1]
    rel_labels = pred_i["rel_labels"]
    tmp_node_relation_node = []
    tmp_node_relation_node_true_label = []
    for j in range(len(rel_pairs1)):
        tmp_node_relation_node.append([rel_pairs0[j], rel_labels[j], rel_pairs1[j]])
        # 给第二个定义为判断是否为truelabel
        tmp_node_relation_node_true_label.append(true_label[rel_pairs1[j]])
    # node_relation_node.append(tmp_node_relation_node)
    pred[str(i)]["node_relation_node"] = tmp_node_relation_node
    tmp_node_relation_node_len =  len(tmp_node_relation_node)
    link_pairs = [[],[]]
    for k in range(tmp_node_relation_node_len):
        for q in range(k+1, tmp_node_relation_node_len):
            # 这里没有将线分类 例如对撞，顺序，等
            if(tmp_node_relation_node[k][0]== tmp_node_relation_node[q][2]):
                link_pairs[0].append(k)
                link_pairs[1].append(q)
            elif(tmp_node_relation_node[k][2]== tmp_node_relation_node[q][0]):
                link_pairs[0].append(q)
                link_pairs[1].append(k)
    pred[str(i)]["link_pairs"] = link_pairs
    node_relation_node_true_label.append(tmp_node_relation_node_true_label)
info["node_relation_node_true_label"] = node_relation_node_true_label

with open(output_info_dir, "w")as f:
    json.dump(info, f)
with open(output_pred_dir, "w")as f:
    json.dump(pred, f)