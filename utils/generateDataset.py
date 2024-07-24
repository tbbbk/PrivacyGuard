import pandas as pd
import json
from tqdm import tqdm
import cv2
import numpy as np
from datetime import datetime


def get_current_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


def find_max_iou_box(box, unique_boxes):
    max_iou = 0
    max_iou_index = -1
    for i, unique_box in enumerate(unique_boxes):
        iou = calculate_iou(box, unique_box)
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i
    return max_iou_index


def find_max_ciou_box(box, unique_boxes):
    max_iou = -2
    max_iou_index = -1
    for i, unique_box in enumerate(unique_boxes):
        iou = calculate_ciou(box, unique_box)
        if iou > max_iou:
            max_iou = iou
            max_iou_index = i
    return max_iou_index


def load_privacy_boxes(txt_path, w, h):
    results = []
    with open(txt_path, 'r') as file:
        for line in file:
            numbers = line.split()
            numbers = [float(num) for num in numbers]
            numbers[3], numbers[4] = numbers[1] + numbers[3], numbers[2] + numbers[4]
            results.append([numbers[1] * w, numbers[2] * h, numbers[3] * w, numbers[4] * h])
    return results


def load_json(json_path):
    with open(json_path, 'r') as f:
        # 加载JSON数据
        data = json.load(f)
    return data


def calculate_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou


def calculate_ciou(box1, box2):
    # Calculate the IoU
    iou = calculate_iou(box1, box2)

    # Calculate the central point distance
    c_box1 = ((box1[2]+box1[0])/2, (box1[3]+box1[1])/2)
    c_box2 = ((box2[2]+box2[0])/2, (box2[3]+box2[1])/2)
    c_dist = np.sqrt((c_box1[0]-c_box2[0])**2 + (c_box1[1]-c_box2[1])**2)

    # Find the width and height of the smallest enclosing box that contains both bounding boxes
    enclose_bbox = [min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])]
    enclose_wh = np.array([enclose_bbox[2] - enclose_bbox[0], enclose_bbox[3] - enclose_bbox[1]])

    # Calculate the diagonal length of the smallest enclosing box that contains both bounding boxes
    c_diag = np.sqrt(enclose_wh[0]**2 + enclose_wh[1]**2)

    # Calculate the aspect ratio
    v = 4 / (np.pi**2) * ((np.arctan((box2[3]-box2[1]) / (box2[2]-box2[0])) - np.arctan((box1[3]-box1[1]) / (box1[2]-box1[0])))**2)

    # Calculate alpha
    alpha = v / (1 - iou + v)

    # Calculate CIoU
    ciou_term = iou - (c_dist / c_diag) - alpha * v

    return ciou_term


"""
数据对齐 csv -> json
"""
def align_raw_data(csv_path, output_custom_data_info_path, output_custom_prediction_path):
    print(f"{get_current_time()} 数据集对齐中")
    idx = 0
    data = pd.read_csv(csv_path)
    result_data_info = {
        "idx_to_files": [],
        "ind_to_classes": [],
        "ind_to_predicates": [],
        "all_true_label": []
    }
    
    result_prediction = {}
    
    result_data_info["ind_to_classes"] = [
            "__background__",
            "airplane",
            "animal",
            "arm",
            "bag",
            "banana",
            "basket",
            "beach",
            "bear",
            "bed",
            "bench",
            "bike",
            "bird",
            "board",
            "boat",
            "book",
            "boot",
            "bottle",
            "bowl",
            "box",
            "boy",
            "branch",
            "building",
            "bus",
            "cabinet",
            "cap",
            "car",
            "cat",
            "chair",
            "child",
            "clock",
            "coat",
            "counter",
            "cow",
            "cup",
            "curtain",
            "desk",
            "dog",
            "door",
            "drawer",
            "ear",
            "elephant",
            "engine",
            "eye",
            "face",
            "fence",
            "finger",
            "flag",
            "flower",
            "food",
            "fork",
            "fruit",
            "giraffe",
            "girl",
            "glass",
            "glove",
            "guy",
            "hair",
            "hand",
            "handle",
            "hat",
            "head",
            "helmet",
            "hill",
            "horse",
            "house",
            "jacket",
            "jean",
            "kid",
            "kite",
            "lady",
            "lamp",
            "laptop",
            "leaf",
            "leg",
            "letter",
            "light",
            "logo",
            "man",
            "men",
            "motorcycle",
            "mountain",
            "mouth",
            "neck",
            "nose",
            "number",
            "orange",
            "pant",
            "paper",
            "paw",
            "people",
            "person",
            "phone",
            "pillow",
            "pizza",
            "plane",
            "plant",
            "plate",
            "player",
            "pole",
            "post",
            "pot",
            "racket",
            "railing",
            "rock",
            "roof",
            "room",
            "screen",
            "seat",
            "sheep",
            "shelf",
            "shirt",
            "shoe",
            "short",
            "sidewalk",
            "sign",
            "sink",
            "skateboard",
            "ski",
            "skier",
            "sneaker",
            "snow",
            "sock",
            "stand",
            "street",
            "surfboard",
            "table",
            "tail",
            "tie",
            "tile",
            "tire",
            "toilet",
            "towel",
            "tower",
            "track",
            "train",
            "tree",
            "truck",
            "trunk",
            "umbrella",
            "vase",
            "vegetable",
            "vehicle",
            "wave",
            "wheel",
            "window",
            "windshield",
            "wing",
            "wire",
            "woman",
            "zebra"
        ],
    result_data_info["ind_to_predicates"] = [
            "__background__",
            "above",
            "across",
            "against",
            "along",
            "and",
            "at",
            "attached to",
            "behind",
            "belonging to",
            "between",
            "carrying",
            "covered in",
            "covering",
            "eating",
            "flying in",
            "for",
            "from",
            "growing on",
            "hanging from",
            "has",
            "holding",
            "in",
            "in front of",
            "laying on",
            "looking at",
            "lying on",
            "made of",
            "mounted on",
            "near",
            "of",
            "on",
            "on back of",
            "over",
            "painted on",
            "parked on",
            "part of",
            "playing",
            "riding",
            "says",
            "sitting on",
            "standing on",
            "to",
            "under",
            "using",
            "walking in",
            "walking on",
            "watching",
            "wearing",
            "wears",
            "with"
        ]
    
    for graph_id in tqdm(data["graph_id"].unique()):
        result_prediction[str(idx)] = {
            "bbox": [],
            "bbox_labels": [],
            "bbox_scores": [],
            "rel_pairs": [[], []],
            "rel_labels": [],
            "rel_scores": []
        }
        
        graph_data = data[data["graph_id"] == graph_id]
        
        # 选出节点
        boxes, unique_boxes = [], []
        types, unique_types = [], []
        for box in graph_data[["start_node", "sxmin", "symin", "sxmax", "symax"]].values.tolist():
            boxes.append(box[1:])
            types.append(box[0])
        for box in graph_data[["end_node", "oxmin", "oymin", "oxmax", "oymax"]].values.tolist():
            boxes.append(box[1:])
            types.append(box[0])
        for i, box in enumerate(boxes):
            if any(calculate_iou(box, prev_box) > 0.9 for prev_box in boxes[:i]): continue 
            unique_boxes.append(box)
            # try:
            unique_types.append(int(types[i]))
            # except MemoryError:
            #      import pdb; pdb.set_trace()
        # 处理prediction
        for box in unique_boxes:
            result_prediction[str(idx)]["bbox"].append(box)
        result_prediction[str(idx)]["bbox_labels"] = unique_types
        result_prediction[str(idx)]["bbox_scores"] = [1 for _ in unique_boxes]  # 置信度设置为1
        for _, row in graph_data.iterrows():
            sbox = [row["sxmin"], row["symin"], row["sxmax"], row["symax"]]
            obox = [row["oxmin"], row["oymin"], row["oxmax"], row["oymax"]]
            result_prediction[str(idx)]["rel_pairs"][0].append(find_max_iou_box(sbox, unique_boxes))  # 起始节点列表
            result_prediction[str(idx)]["rel_pairs"][1].append(find_max_iou_box(obox, unique_boxes))  # 结束节点列表
            result_prediction[str(idx)]["rel_labels"].append(row["relation"])
            result_prediction[str(idx)]["rel_scores"].append(1)
        
        labels = [0 for _ in range(len(unique_boxes))]
        result_data_info["all_true_label"].append(labels)
        result_data_info["idx_to_files"].append(graph_data["path"].iloc[0])
        
        idx += 1
        
    with open(output_custom_data_info_path, 'w') as f:
            json.dump(result_data_info, f)
            
    with open(output_custom_prediction_path, 'w') as f:
            json.dump(result_prediction, f)
    print(f"{get_current_time()} 数据集已对齐")


"""
找最近邻并将其标注为1
"""
def merge_privacy_obj_into_sgg(custom_data_info_path, custom_prediction_path):
    print(f"{get_current_time()} 正在融合隐私物体到sgg")
    custom_data_info = load_json(custom_data_info_path)
    custom_prediction = load_json(custom_prediction_path)
    for idx, image_path in tqdm(enumerate(custom_data_info['idx_to_files'])):
        img = cv2.imread(image_path)
        # print(type(img))
        # import pdb; pdb.set_trace()
        height, width, _ = img.shape
        txt_path = image_path.replace('img', 'txt').replace('jpg', 'txt')
        privacy_boxes = load_privacy_boxes(txt_path, height, width)
        boxes = custom_prediction[str(idx)]['bbox']
        for privacy_box in privacy_boxes:
            index = find_max_ciou_box(privacy_box, boxes)
            custom_data_info['all_true_label'][idx][index] = 1
    with open(custom_data_info_path, 'w') as f:
            json.dump(custom_data_info, f)
    print(f"{get_current_time()} 融合完毕")
            

"""
平衡节点
"""
def balance_nodes(custom_data_info_path, custom_prediction_path):
    print(f"{get_current_time()} 正在平衡隐私节点")
    custom_data_info = load_json(custom_data_info_path)
    custom_prediction = load_json(custom_prediction_path)
    for idx, labels in tqdm(enumerate(custom_data_info['all_true_label'])):
        privacy_indices = [i for i, x in enumerate(labels) if x == 1]
        for index in privacy_indices:
            custom_data_info['all_true_label'][idx].append(1)
            custom_prediction[str(idx)]['bbox'].append(custom_prediction[str(idx)]['bbox'][index])
            custom_prediction[str(idx)]['bbox_labels'].append(custom_prediction[str(idx)]['bbox_labels'][index])
            custom_prediction[str(idx)]['bbox_scores'].append(custom_prediction[str(idx)]['bbox_scores'][index])
            for relation_index in range(len(custom_prediction[str(idx)]['rel_pairs'][0])):
                if custom_prediction[str(idx)]['rel_pairs'][0][relation_index] == index: 
                    custom_prediction[str(idx)]['rel_pairs'][0].append(len(labels) - 1)
                    custom_prediction[str(idx)]['rel_pairs'][1].append(custom_prediction[str(idx)]['rel_pairs'][1][relation_index])
                    custom_prediction[str(idx)]['rel_labels'].append(custom_prediction[str(idx)]['rel_labels'][relation_index])
                    custom_prediction[str(idx)]['rel_scores'].append(1)
                elif custom_prediction[str(idx)]['rel_pairs'][1][relation_index] == index:
                    custom_prediction[str(idx)]['rel_pairs'][1].append(len(labels) - 1)
                    custom_prediction[str(idx)]['rel_pairs'][0].append(custom_prediction[str(idx)]['rel_pairs'][0][relation_index])
                    custom_prediction[str(idx)]['rel_labels'].append(custom_prediction[str(idx)]['rel_labels'][relation_index])
                    custom_prediction[str(idx)]['rel_scores'].append(1)
    with open(custom_data_info_path, 'w') as f:
            json.dump(custom_data_info, f)
            
    with open(custom_prediction_path, 'w') as f:
            json.dump(custom_prediction, f)
    print(f"{get_current_time()} 隐私节点平衡完毕")
        

if __name__ == "__main__":
    custom_data_info_path = '/home/zhuohang/HGR/test/reltr_custom_data_info.json'
    custom_prediction_path = '/home/zhuohang/HGR/test/reltr_custom_prediction.json'
    align_raw_data(csv_path='/mnt/data0/zhuohang/reltr/mosaic.csv', 
                   output_custom_data_info_path=custom_data_info_path,
                   output_custom_prediction_path=custom_prediction_path)
    merge_privacy_obj_into_sgg(custom_data_info_path=custom_data_info_path,
                               custom_prediction_path=custom_prediction_path)
    balance_nodes(custom_data_info_path=custom_data_info_path, 
                  custom_prediction_path=custom_prediction_path)
    balance_nodes(custom_data_info_path=custom_data_info_path, 
                  custom_prediction_path=custom_prediction_path)
    balance_nodes(custom_data_info_path=custom_data_info_path, 
                  custom_prediction_path=custom_prediction_path)