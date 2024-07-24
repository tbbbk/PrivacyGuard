# blur equals to some nodes' categories are wrong
import sys
import yaml
import time
import numpy as np
from tqdm import tqdm
import os
with open("./configs/model.yaml", "r") as f:
    model_yaml = yaml.safe_load(f)

project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)

import utils.load_data as ld
import random as rand
import json

mosaic_pred = ld.load_mosaic_prediction(file_name="random_balance_custom_prediction.json")
mosaic_info = ld.load_mosaic_info(file_name="random_balance_custom_data_info.json")
privacy_pred = ld.load_privacy_prediction(file_name="random_balance_custom_prediction.json")
privacy_info = ld.load_privacy_info(file_name="random_balance_custom_data_info.json")
blur_mosaic_pred_name = "blur_random_balance_custom_prediction.json"
blur_privacy_pred_name = "blur_random_balance_custom_prediction.json"
mosaic_save_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic"
privacy_save_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy"

category_num = len(mosaic_info["ind_to_classes"]) - 1
num_rate = 2
# 1/10 is wrong
for i in tqdm(range(len(mosaic_pred))):
    node_categroy = mosaic_pred[str(i)]["bbox_labels"]
    for j in range(len(node_categroy)):
        if(rand.randint(0,num_rate) != 0):
            node_categroy[j] = rand.randint(0, category_num)
    mosaic_pred[str(i)]["bbox_labels"] = node_categroy

with open(f"{mosaic_save_dir}/{blur_mosaic_pred_name}", "w") as json_file:
    json.dump(mosaic_pred, json_file)



for i in tqdm(range(len(privacy_pred))):
    node_categroy = privacy_pred[str(i)]["bbox_labels"]
    for j in range(len(node_categroy)):
        if(rand.randint(0,num_rate) == 0):
            node_categroy[j] = rand.randint(0,category_num)
    privacy_pred[str(i)]["bbox_labels"] = node_categroy

with open(f"{privacy_save_dir}/{blur_privacy_pred_name}", "w") as json_file:
    json.dump(privacy_pred, json_file)