import json
import random
from tqdm import tqdm
# 因为balance后的数据集存在后面都是1，前面都是0的问题，所以我们再打乱一下就行

def generate_shuffled_list(n):
    original_list = list(range(n))  # 生成0到n-1的有序列表
    shuffled_list = random.sample(
        original_list, len(original_list)
    )  # 使用random.sample函数打乱列表
    return shuffled_list


# n = 10  # 你可以替换为你想要的n的值
# shuffled_list = generate_shuffled_list(n)
# print(shuffled_list)


# pred_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/balance_custom_prediction.json"
# info_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/balance_custom_data_info.json"
pred_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/balance_custom_prediction.json"
info_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/balance_custom_data_info.json"


with open(pred_dir, "r") as pred_reader:
    pred = json.load(pred_reader)

with open(info_dir, "r") as info_reader:
    info = json.load(info_reader)

new_pred = {}
new_info = {}
new_info["idx_to_files"] = info["idx_to_files"]
new_info["ind_to_classes"] = info["ind_to_classes"]
new_info["ind_to_predicates"] = info["ind_to_predicates"]
new_info["all_true_label"] = []
new_all_true_label = []
img_num = len(info["idx_to_files"])
# print(img_num)

for i in tqdm(range(img_num)):
    node_num = len(info["all_true_label"][i])
    # print(node_num)
    shuffled_list = generate_shuffled_list(node_num)
    # print(shuffled_list)
    old_node_pred = pred[str(i)]
    old_true_label = info["all_true_label"][i]

    tmp_true_label = []
    new_tmp_pred = {"bbox": [], "bbox_labels": [], "bbox_scores": []}
    tmp_rel_pairs = [[], []]
    # bbox, bbox_labels, bbox_scores, rel_pairs[[],[]], rel_labels, rel_scores
    for node_i in range(node_num):
        # [24, 19, 17, 6, 7, 3, 14, 13, 1, 2, 5, 15, 22, 11, 23, 9, 10, 20, 16, 4, 12, 8, 18, 0, 21]
        # 这是顺序，24代表原来处在24的位置
        # print(f"{node_i} ---> {shuffled_list[node_i]}")
        # process info true label
        tmp_true_label.append(old_true_label[shuffled_list[node_i]])
        # process pred
        new_tmp_pred["bbox"].append(old_node_pred["bbox"][shuffled_list[node_i]])
        new_tmp_pred["bbox_labels"].append(
            old_node_pred["bbox_labels"][shuffled_list[node_i]]
        )
        new_tmp_pred["bbox_scores"].append(
            old_node_pred["bbox_scores"][shuffled_list[node_i]]
        )
        # [[3,2,1],[4,5,0]]  [0,5,3,4,2,1] ==> [[2,4,5],[3,1,0]]
        # 还是靠位置
    rela_num = len(old_node_pred["rel_labels"])
    for rela_i in range(rela_num):
        tmp_rel_pairs[0].append(
            shuffled_list.index(old_node_pred["rel_pairs"][0][rela_i])
        )
        tmp_rel_pairs[1].append(
            shuffled_list.index(old_node_pred["rel_pairs"][1][rela_i])
        )
    new_tmp_pred["rel_pairs"] = tmp_rel_pairs
    new_tmp_pred["rel_labels"] = old_node_pred["rel_labels"]
    new_tmp_pred["rel_scores"] = old_node_pred["rel_scores"]
    new_info["all_true_label"].append(tmp_true_label)
    new_pred[str(i)] = new_tmp_pred

# save_pred_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/random_balance_custom_prediction.json"
# save_info_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/random_balance_custom_data_info.json"
save_pred_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/random_balance_custom_prediction.json"
save_info_dir = "/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/random_balance_custom_data_info.json"


with open(save_pred_dir, "w") as f:
    json.dump(new_pred, f)

with open(save_info_dir, "w") as f:
    json.dump(new_info, f)
