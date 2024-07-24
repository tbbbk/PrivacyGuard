import json
import yaml
import sys
import cv2
with open("./configs/model.yaml", "r")as f:
    model_yaml = yaml.safe_load(f)
project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)
import utils.load_data as ld
import torch
from Model.model import GCN, privacy_dataset, EarlyStopping
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Subset

mosaic_info = ld.load_mosaic_info("all")
mosaic_predict = ld.load_mosaic_prediction("all")


pth_dir = "./pth/model_900.pth"
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device:{device}")
dataset_name = "mosaic"
model_batch_size = 1

# 创建模型
model = GCN(in_channels=input_layer_dim, hidden_channels=hidden_layer_dim, out_channels=output_layer_dim).to(device)


# 创建数据加载器
def create_data_loader(data_list, batch_size, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


full_dataset = privacy_dataset(dataset_name)
    # 创建数据加载器
# data_loader = create_data_loader(full_dataset, batch_size=model_batch_size, shuffle=True)

# print(full_dataset[0]["x"])
img_num = 2
# x = full_dataset[img_num]["x"]
# y = full_dataset[img_num]["y"]
# edge_index = full_dataset[img_num]["edge_index"]
# link_embed = full_dataset[img_num]["link_embed"]

input_data = full_dataset[img_num]

model.load_state_dict(torch.load(pth_dir))
model.eval()

input_data = input_data.to(device)
output = model(input_data)
# print(output)
pred = output.argmax(dim=1)
print("pred:")
print(pred)
print("true:")
print(mosaic_info["all_true_label"][img_num])

img_dir = mosaic_info["idx_to_files"][img_num]
# 读取图片
image = cv2.imread(img_dir)
# for i,p in enumerate(mosaic_info["all_true_label"][img_num]):
for i,p in enumerate(pred):
    if(p == 1):
        x1, y1, x2, y2 = mosaic_predict[str(img_num)]["bbox"][i]  # 目标位置坐标
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
# 显示图片
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()