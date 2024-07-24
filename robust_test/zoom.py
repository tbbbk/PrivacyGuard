# zoom equals to some nodes are missing
import sys
import yaml
import random as rand
import time
import numpy as np
from tqdm import tqdm
import os
with open("./configs/model.yaml", "r") as f:
    model_yaml = yaml.safe_load(f)

project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)

import utils.load_data as ld
import json

mosaic_pred = ld.load_mosaic_prediction(file_name="random_balance_custom_prediction.json")
mosaic_info = ld.load_mosaic_info(file_name="random_balance_custom_data_info.json")
privacy_pred = ld.load_privacy_prediction(file_name="random_balance_custom_prediction.json")
privacy_info = ld.load_privacy_info(file_name="random_balance_custom_data_info.json")
zoom_mosaic_pred_name = "zoom_random_balance_custom_prediction.json"
zoom_privacy_pred_name = "zoom_random_balance_custom_prediction.json"
mosaic_save_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic"
privacy_save_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy"


num_rate = 2
# 1/10 is wrong
for i in tqdm(range(len(mosaic_pred))):
    node_num = len(mosaic_pred[str(i)]["rel_labels"])
    relation_node_1 = mosaic_pred[str(i)]["rel_pairs"][0]
    relation_node_2 = mosaic_pred[str(i)]["rel_pairs"][1]
    link_embed = mosaic_pred[str(i)]["rel_labels"]
    link_score = mosaic_pred[str(i)]["rel_scores"]
    new_relation_node_1 = []
    new_relation_node_2 = []
    new_link_embed = []
    new_link_score = []
    for j in range(len(relation_node_1)):
        if(rand.randint(0,num_rate) == 0):
            new_relation_node_1.append(relation_node_1[j])
            new_relation_node_2.append(relation_node_2[j])
            new_link_embed.append(link_embed[j])
            new_link_score = [link_score[j]]
    mosaic_pred[str(i)]["rel_pairs"] = [new_relation_node_1, new_relation_node_2]
    mosaic_pred[str(i)]["rel_labels"] = new_link_embed
    mosaic_pred[str(i)]["rel_scores"] = new_link_score

with open(f"{mosaic_save_dir}/{zoom_mosaic_pred_name}", "w") as json_file:
    json.dump(mosaic_pred, json_file)


for i in tqdm(range(len(privacy_pred))):
    node_num = len(privacy_pred[str(i)]["rel_labels"])
    relation_node_1 = privacy_pred[str(i)]["rel_pairs"][0]
    relation_node_2 = privacy_pred[str(i)]["rel_pairs"][1]
    link_embed = privacy_pred[str(i)]["rel_labels"]
    link_score = privacy_pred[str(i)]["rel_scores"]
    new_relation_node_1 = []
    new_relation_node_2 = []
    new_link_embed = []
    new_link_score = []
    for j in range(len(relation_node_1)):
        if(rand.randint(0,num_rate) != 0):
            new_relation_node_1.append(relation_node_1[j])
            new_relation_node_2.append(relation_node_2[j])
            new_link_embed.append(link_embed[j])
            new_link_score = [link_score[j]]
    privacy_pred[str(i)]["rel_pairs"] = [new_relation_node_1, new_relation_node_2]
    privacy_pred[str(i)]["rel_labels"] = new_link_embed
    privacy_pred[str(i)]["rel_scores"] = new_link_score

with open(f"{privacy_save_dir}/{zoom_privacy_pred_name}", "w") as json_file:
    json.dump(privacy_pred, json_file)
