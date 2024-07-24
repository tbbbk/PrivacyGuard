import random
import numpy as np
import yaml
import sys
with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import utils.load_data as ld
from tqdm import tqdm
import json
with open("/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/custom_prediction.json","r") as f:
    prediction = json.load(f)


for key, data in tqdm(prediction.items()):
    # print(data)
    link_info = data["rel_pairs"]
    new_lin_info = [list(row) for row in zip(*link_info)]
    # data["rel_pairs"] = new_lin_info
    prediction[f"{key}"]["rel_pairs"]= new_lin_info
print("saving................")
with open("/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/new_custom_prediction.json", "w") as f:
    json.dump(prediction, f)
print("saved")
    