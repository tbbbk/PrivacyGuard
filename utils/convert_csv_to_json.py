import pandas as pd
import json


def convert2data(csv_path, outpath):
    # 读取CSV文件
    # data = pd.read_csv('/mnt/data0/zhuohang/reltr/mosaic.csv')
    data = pd.read_csv(csv_path)

    # 初始化空的字典来保存最终的结果
    result = {
        "idx_to_files": [],
        "ind_to_classes": [],
        "ind_to_predicates": [],
        "all_true_label": []
    }

    # 填充idx_to_files

    # 填充ind_to_classes和ind_to_predicates
    # 这里假设你的CSV文件中有一个名为"classes"的列和一个名为"predicates"的列
    # 如果实际情况不同，你需要修改这部分代码
    result["ind_to_classes"] = [
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
    result["ind_to_predicates"] = [
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

    # 填充all_true_label
    # 这里假设你的CSV文件中有一个名为"label"的列，且该列的值是一个列表
    # 如果实际情况不同，你需要修改这部分代码
    for graph_id in data["graph_id"].unique():
        graph_data = data[data["graph_id"] == graph_id]
        labels = [1 if gt == 'privacy' else 0 for gt in graph_data["graph_type"]]
        result["all_true_label"].append(labels)
        result["idx_to_files"].append(graph_data["path"].iloc[0])

    # 将结果保存为JSON文件
    # with open('/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/reltr_custom_data_info.json', 'w') as f:
    with open(outpath, 'w') as f:
        json.dump(result, f)
        
        
def convert2prediction(csv_path, outpath):
    import pandas as pd
    import json

    # 读取CSV文件
    # data = pd.read_csv('/mnt/data0/zhuohang/reltr/mosaic.csv')
    data = pd.read_csv(csv_path)

    # 初始化空的字典来保存最终的结果
    result = {}

    # 获取所有独特的graph_id，并按照它们在CSV文件中出现的顺序进行排序
    unique_graph_ids = data["graph_id"].unique()

    # 遍历每一个graph_id
    for graph_id in unique_graph_ids:
        graph_data = data[data["graph_id"] == graph_id]
        
        # 初始化每个graph_id的字典
        result[str(graph_id)] = {
            "bbox": [],
            "bbox_labels": [],
            "bbox_scores": [],
            "rel_pairs": [[], []],  # 修改为两个列表
            "rel_labels": [],
            "rel_scores": []
        }

        # 填充bbox, bbox_labels, bbox_scores
        for _, row in graph_data.iterrows():
            result[str(graph_id)]["bbox"].append([row["sxmin"], row["symin"], row["sxmax"], row["symax"]])
            result[str(graph_id)]["bbox"].append([row["oxmin"], row["oymin"], row["oxmax"], row["oymax"]])
            result[str(graph_id)]["bbox_labels"].append(row["start_node"])
            result[str(graph_id)]["bbox_labels"].append(row["end_node"])
            result[str(graph_id)]["bbox_scores"].extend([1, 1])  # 置信度设置为1

        # 填充rel_pairs, rel_labels, rel_scores
        for _, row in graph_data.iterrows():
            result[str(graph_id)]["rel_pairs"][0].append(row["start_node"])  # 起始节点列表
            result[str(graph_id)]["rel_pairs"][1].append(row["end_node"])  # 结束节点列表
            result[str(graph_id)]["rel_labels"].append(row["relation"])
            result[str(graph_id)]["rel_scores"].append(1)  # 置信度设置为1

    # 将结果保存为JSON文件
    with open(outpath, 'w') as f:
    # with open('/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic/reltr_custom_prediction.json', 'w') as f:
        json.dump(result, f)


if __name__  == '__main__':
    # privacy就把csv的名字改一下
    # 这里建议重新生成一下，我搞了一般半有点乱了
    # 这个生成prediction
    convert2prediction(csv_path='/mnt/data0/zhuohang/reltr/mosaic.csv', outpath='/home/zhuohang/HGR/test/reltr_custom_prediction.json')
    # 这个生成custom data info
    convert2data(csv_path='/mnt/data0/zhuohang/reltr/mosaic.csv', outpath='/home/zhuohang/HGR/test/reltr_custom_data_info.json')