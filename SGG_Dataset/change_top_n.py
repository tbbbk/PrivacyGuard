import json
import yaml
from tqdm import tqdm
input_info_dir = "/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/old_custom_data_info.json"
input_pred_dir = "/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/old_custom_prediction.json"
output_info_dir = "/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/custom_data_info.json"
output_pred_dir = "/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/mosaic/custom_prediction.json"
with open('./configs/model.yaml') as file:
    model_info = yaml.safe_load(file)

top_n = model_info["model"]["box_topk"]
top_link_pred = model_info["model"]["relationship"]["trust_val"]
save_pred = {}
with open(input_pred_dir, "r")as f:
    data = json.load(f)
for n,pred in tqdm(enumerate(data.items())):
    print(n)
    # print(data[str(n)])
    top_bbox = data[f"{n}"]["bbox"][:top_n]
    top_bbox_labels = data[f"{n}"]["bbox_labels"][:top_n]
    top_bbox_scores = data[f"{n}"]["bbox_scores"][:top_n]
    rel_pairs = data[f"{n}"]["rel_pairs"]
    rel_labels = data[f"{n}"]["rel_labels"]
    rel_scores = data[f"{n}"]["rel_scores"]
    # print(rel_pairs)
    # print(rel_pairs[1][0]) 
    top_rel_pairs0 = []
    top_rel_pairs1 = []
    top_rel_labels = []
    top_rel_scores = []
    node_len = len(rel_scores)
    for i in range(node_len):
        if(rel_pairs[0][i] < top_n and rel_pairs[1][i] < top_n and rel_scores[i] > top_link_pred):
            top_rel_pairs0.append(rel_pairs[0][i])
            top_rel_pairs1.append(rel_pairs[1][i])
            top_rel_labels.append(rel_labels[i])
            top_rel_scores.append(rel_scores[i])
    
    save_pred[str(n)] = {"bbox": top_bbox, "bbox_labels": top_bbox_labels, "bbox_scores": top_bbox_scores, "rel_pairs":[top_rel_pairs0, top_rel_pairs1],
                         "rel_labels": top_rel_labels,"rel_scores": top_rel_scores}
    
with open(output_pred_dir, "w") as f:
    json.dump(save_pred, f)


with open(input_info_dir, "r")as f:
    data = json.load(f)

# idx_to_files = data["idx_to_files"]
# ind_to_classes = data["ind_to_classes"]
# ind_to_predicates = data["ind_to_predicates"]
img_len = len(data["all_true_label"])
for i in tqdm(range(img_len)):
    data["all_true_label"][i] = data["all_true_label"][i][:top_n]

with open(output_info_dir, "w") as f:
    json.dump(data, f)