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

def process_privacy_dataset_1000_privacy(box_topk = 30, output_file_name="custom_data_info.json"):
    # privacy_dataset_1000
    save_dir ="/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy"
    privacy_imgs_info = ld.load_privacy_info("all", file_name="origin_custom_data_info.json")
    privacy_imgs_dir = privacy_imgs_info["idx_to_files"]
    privacy_predict_info = ld.load_privacy_prediction("all", file_name="custom_prediction.json")
    privacy_obj_class = privacy_imgs_info["ind_to_classes"]
    all_true_label = []
    print(len(privacy_imgs_dir))
    for i, img_dir in tqdm(enumerate(privacy_imgs_dir)):

        # true_label = [0] * box_topk   
        bbox = privacy_predict_info.get(str(i))
        if bbox is None:
            continue
        node_topk_num = min(box_topk, len(privacy_predict_info[str(i)]["bbox_labels"]))
        true_label = [0] * node_topk_num 
        txt_dir = img_dir.replace("img", "txt")
        txt_dir = txt_dir.replace("jpg", "txt")

        prediction = []
        img_data = cv2.imread(img_dir)
        height, width, _ = img_data.shape

        with open(txt_dir, "r") as f:
            lines = f.readlines()

        values = []
        for line in lines:
            line_values = line.split()
            if len(line_values) >= 4:
                values.extend([float(val) for val in line_values[-4:]])
                if len(values) == 4:
                    prediction.append(values)
                    values = []

        bbox = privacy_predict_info[str(i)]['bbox'][:box_topk]
        box_labels = privacy_predict_info[str(i)]['bbox_labels'][:box_topk]

        for pred in prediction:
            x, y, w, h = pred
            x1 = x * width
            y1 = y * height
            x2 = x1 + w * width
            y2 = y1 + h * height
            my_box = [x1, y1, x2, y2]

            # 计算与my_box最接近的边界框
            min_distance = math.inf
            nearest_box_index = None

            for i, box in enumerate(bbox):
                bx1, by1, bx2, by2 = box
                box_center_x = (bx1 + bx2) / 2
                box_center_y = (by1 + by2) / 2
                distance = math.sqrt((box_center_x - (my_box[0] + my_box[2]) / 2)**2 +
                                    (box_center_y - (my_box[1] + my_box[3]) / 2)**2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_box_index = i
            true_label[nearest_box_index] = 1
        all_true_label.append(true_label)
    print(len(all_true_label))
    print(f"Saving at {save_dir}......................")
    privacy_imgs_info["all_true_label"] = all_true_label
    # 将更新后的 JSON 数据保存回文件
    with open(f'{save_dir}/{output_file_name}', 'w') as f:
        json.dump(privacy_imgs_info, f)

def process_pred(input_info_dir, input_pred_dir, output_info_dir, output_pred_dir, top_link_pred = 0.5, top_n = 30, link_mode = "not_changed"):
    save_pred = {}
    with open(input_pred_dir, "r")as f:
        data = json.load(f)
    if(link_mode == "not_changed"):
        # rel_pairs=[[1,6],[2,7],[3,8],[4,9],[5,10]]
        for n,pred in tqdm(enumerate(data.items())):
            # print(n)
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
            # print(f"node_len = {node_len}")
            # print(f"pair_len = {len(rel_pairs)}")
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
        


def balance_dataset_add(true_list:list, data_list:list, data_list2:list, data_list3:list):
    # 提取随机的节点，以及随机的 top_bbox, top_bbox_labels, top_bbox_scores, rand_list

    rand_list = []
    for i in range(len(data_list) - 2 * len(true_list)):
        # print(true_list)
        # print(len(true_list))
        if len(true_list) == 0:
            continue
        ran = random.randint(0, len(true_list)-1)
        data_list.append(data_list[true_list[ran]])
        data_list2.append(data_list2[true_list[ran]])
        data_list3.append(data_list3[true_list[ran]])
        rand_list.append(true_list[ran])
    return data_list, data_list2, data_list3, rand_list


def balance_dataset_add_SMOTE(true_list:list, data_list:list, data_list2:list, data_list3:list):
    # 提取随机的节点，以及随机的 top_bbox, top_bbox_labels, top_bbox_scores, rand_list

    rand_list = []
    for i in range(len(data_list) - 2 * len(true_list)):
        # print(true_list)
        # print(len(true_list))
        if len(true_list) == 0:
            continue
        ran = random.randint(0, len(true_list)-1)
        ran2 = random.randint(0, len(true_list)-1)
        data_list.append(data_list[true_list[ran2]])
        data_list2.append(data_list2[true_list[ran2]])
        data_list3.append(data_list3[true_list[ran]])
        rand_list.append(true_list[ran])
    return data_list, data_list2, data_list3, rand_list

def balance_dataset(input_info_dir, input_pred_dir, output_info_dir, output_pred_dir):
    with open(input_info_dir, "r")as f:
        data = json.load(f)
    img_len = len(data["all_true_label"])
    print(img_len)
    print("processing balance dataset:")
    all_true_label = []
    for i in tqdm(range(img_len)):
        true_label = []
        for item, is_true in enumerate(data["all_true_label"][i]):
            if(is_true == 1):
                true_label.append(item)
        all_true_label.append(true_label)

        if(sum(true_label) > 0):
            data["all_true_label"][i] = data["all_true_label"][i] + [1] * (len(data["all_true_label"][i]) - 2 * len(true_label))
    # print(all_true_label)
    with open(output_info_dir, "w") as f:
        json.dump(data, f)



    save_pred = {}
    with open(input_pred_dir, "r")as f:
        data = json.load(f)
    # data.items() == 1000
    # for n,pred in tqdm(enumerate(data.items())):
    for n in tqdm(range(949)):
            # TODO !!??
            # print(n)
            # TODO 加个节点打乱算法 在另外一个文件夹
            true_label = all_true_label[n]
            if(sum(true_label) > 0):    
                bbox = data.get(str(n))
                if bbox is None:
                    continue
                top_bbox, top_bbox_labels, top_bbox_scores, rand_list= balance_dataset_add(true_label, data[f"{n}"]["bbox"], data[f"{n}"]["bbox_labels"], data[f"{n}"]["bbox_scores"])
                # rand_list = [7,6,7,7,7,6] 就是第top以后每个节点是什么节点的复制
                # 增加节点的关系
                rel_pairs = data[f"{n}"]["rel_pairs"]
                rel_labels = data[f"{n}"]["rel_labels"]
                rel_scores = data[f"{n}"]["rel_scores"]
                # print(rel_pairs)
                # print(rel_pairs[1][0]) 
                top_rel_pairs0 = rel_pairs[0]
                top_rel_pairs1 = rel_pairs[1]
                top_rel_labels = rel_labels
                top_rel_scores = rel_scores
                node_len = len(rel_scores)
                add_len = len(rand_list)
                # print(f"node_len = {node_len}")
                # print(f"pair_len = {len(rel_pairs[0])}")
                for i in range(node_len, node_len + add_len):
                    for origin in range(node_len):
                        
                        # 这里没写完！！循环查询 应该写完了，检查一下
                        if(top_rel_pairs0[origin] == i):
                            # print("yes")
                            top_rel_pairs0.append(i)
                            top_rel_pairs1.append(rel_pairs[1][origin])
                            top_rel_labels.append(rel_labels[origin])
                            top_rel_scores.append(rel_scores[origin])
                        if(top_rel_pairs1[origin] == i):
                            # print("no")
                            top_rel_pairs1.append(i)
                            top_rel_pairs0.append(rel_pairs[0][origin])
                            top_rel_labels.append(rel_labels[origin])
                            top_rel_scores.append(rel_scores[origin])
                else:
                    top_bbox, top_bbox_labels, top_bbox_scores= data[f"{n}"]["bbox"], data[f"{n}"]["bbox_labels"], data[f"{n}"]["bbox_scores"]
                    rel_pairs = data[f"{n}"]["rel_pairs"]
                    rel_labels = data[f"{n}"]["rel_labels"]
                    rel_scores = data[f"{n}"]["rel_scores"]
                    # print(rel_pairs)
                    # print(rel_pairs[1][0]) 
                    top_rel_pairs0 = rel_pairs[0]
                    top_rel_pairs1 = rel_pairs[1]
                    top_rel_labels = rel_labels
                    top_rel_scores = rel_scores
            save_pred[str(n)] = {"bbox": top_bbox, "bbox_labels": top_bbox_labels, "bbox_scores": top_bbox_scores, "rel_pairs":[top_rel_pairs0, top_rel_pairs1],
                                "rel_labels": top_rel_labels,"rel_scores": top_rel_scores}


    with open(output_pred_dir, "w") as f:
            json.dump(save_pred, f)


def privacy_dataset_1000_privacy_change(box_topk = 30, top_link_pred = 0.5, want_balance = "True"):
    if(want_balance == "True"):
        output_file_name = "custom_data_info.json"
        process_privacy_dataset_1000_privacy(box_topk=box_topk, output_file_name = output_file_name)
        input_info_dir = f"/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/{output_file_name}"
        input_pred_dir = f"/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/custom_prediction.json"
        output_info_dir = f"/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/{output_file_name}"
        output_pred_dir = f"/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy/222balance_custom_prediction.json"
        process_pred(input_info_dir, input_pred_dir, output_info_dir, output_pred_dir, top_link_pred, top_n=box_topk)
        balance_dataset(output_info_dir, output_pred_dir, output_info_dir, output_pred_dir)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="privacy_dataset_1000_privacy",help='the dataset want to change(privacy_dataset_1000_public, privacy_dataset_1000_privacy, mosaic, all)')
    parser.add_argument('--want_balance', default="True",help='True or False')
    args = parser.parse_args()
    dataset_name = args.dataset
    want_bal = args.want_balance
    # print(f"want balance:{want_bal}")
    with open('./configs/model.yaml') as file:
        model_info = yaml.safe_load(file)

    # 想修改数据集，请修改model.yaml,但文件位置请修改上面的函数
    top_n = model_info["model"]["box_topk"]
    top_link_pred = model_info["model"]["relationship"]["trust_val"]
    box_topk = top_n  # 选择前k个边界框
    if(dataset_name == "privacy_dataset_1000_privacy"):
        privacy_dataset_1000_privacy_change(box_topk=box_topk, top_link_pred = top_link_pred, want_balance=want_bal)