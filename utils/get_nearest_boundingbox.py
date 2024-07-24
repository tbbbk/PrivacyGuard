# privacy_dataset_1000
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
from tqdm import tqdm
box_topk = 30  # 选择前k个边界框
save_dir ="/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/privacy"
privacy_imgs_info = ld.load_privacy_info("all")
privacy_imgs_dir = privacy_imgs_info["idx_to_files"]
privacy_predict_info = ld.load_privacy_prediction("all")
privacy_obj_class = privacy_imgs_info["ind_to_classes"]
all_true_label = []

for i, img_dir in tqdm(enumerate(privacy_imgs_dir)):

    true_label = [0] * box_topk 
    
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

    # bbox = privacy_predict_info[str(i)]['bbox'][:box_topk]
    bbox = privacy_predict_info.get(str(i))
    if bbox is not None:
        bbox = bbox['bbox'][:box_topk]
    else:
        continue  # Skip this iteration if the key is not found

    box_labels = privacy_predict_info[str(i)]['bbox_labels'][:box_topk]

    # # 绘制图片和边界框
    # plt.figure()
    # plt.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))

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

        # 输出最接近的边界框索引
        nearest_box_number = nearest_box_index
        # print("最接近的边界框是第", nearest_box_number, "个     ", privacy_obj_class[box_labels[nearest_box_index]])
        true_label[nearest_box_index] = 1
        # # 绘制my_box
        # rect = plt.Rectangle((my_box[0], my_box[1]), my_box[2] - my_box[0], my_box[3] - my_box[1],
        #                     fill=False, edgecolor='b', linewidth=2)
        # plt.gca().add_patch(rect)

        # # 绘制最接近的边界框
        # x1, y1, x2, y2 = bbox[nearest_box_index]
        # rect = plt.Rectangle((int(x1), int(y1)), int(x2) - int(x1), int(y2) - int(y1), fill=False, edgecolor='r', linewidth=2)
        # plt.gca().add_patch(rect)
    

    # # 显示图像和边界框
    # plt.axis('off')
    # plt.show()
    all_true_label.append(true_label)

print(f"Saving at {save_dir}......................")
privacy_imgs_info["all_true_label"] = all_true_label
# 将更新后的 JSON 数据保存回文件
with open(f'{save_dir}/reltr_custom_data_info.json', 'w') as f:
    json.dump(privacy_imgs_info, f)







# # ppublic_dataset_1000
# import math
# import matplotlib.pyplot as plt
# import sys
# import yaml
# with open("./configs/model.yaml", "r")as f:
#     model_yaml = yaml.safe_load(f)
# project_dir = model_yaml["project_dir"]
# sys.path.append(project_dir)
# import utils.load_data as ld
# import cv2
# import json
# from tqdm import tqdm
# box_topk = 30  # 选择前k个边界框
# save_dir ="/media/zhuohang/Disk2/SGG-GCN/datasets/SGG_Dataset/privacy_dataset_1000/public"
# privacy_imgs_info = ld.load_public_info("all")
# privacy_imgs_dir = privacy_imgs_info["idx_to_files"]
# all_true_label = []

# for i, img_dir in tqdm(enumerate(privacy_imgs_dir)):

#     true_label = [0] * box_topk 
    

#     all_true_label.append(true_label)

# print(f"Saving at {save_dir}......................")
# privacy_imgs_info["all_true_label"] = all_true_label
# # 将更新后的 JSON 数据保存回文件
# with open(f'{save_dir}/updated_info.json', 'w') as f:
#     json.dump(privacy_imgs_info, f)



# mosaic

# import math
# import matplotlib.pyplot as plt
# import sys
# import yaml
# with open("./configs/model.yaml", "r")as f:
#     model_yaml = yaml.safe_load(f)
# project_dir = model_yaml["project_dir"]
# sys.path.append(project_dir)
# import utils.load_data as ld
# import cv2
# import json
# from tqdm import tqdm
# box_topk = 30  # 选择前k个边界框
# save_dir ="/home/zhuohang/SGG-GCN/datasets/SGG_Dataset/mosaic"
# privacy_imgs_info = ld.load_mosaic_info("all")
# privacy_imgs_dir = privacy_imgs_info["idx_to_files"]
# privacy_predict_info = ld.load_mosaic_prediction("all")
# privacy_obj_class = privacy_imgs_info["ind_to_classes"]
# all_true_label = []

# for i, img_dir in tqdm(enumerate(privacy_imgs_dir)):

#     true_label = [0] * box_topk 
    
#     txt_dir = img_dir.replace("fakeimg", "txt")
#     txt_dir = txt_dir.replace("jpg", "txt")

#     prediction = []
#     img_data = cv2.imread(img_dir)
#     height, width, _ = img_data.shape

#     with open(txt_dir, "r") as f:
#         lines = f.readlines()

#     values = []
#     for line in lines:
#         line_values = line.split()
#         if len(line_values) >= 4:
#             values.extend([float(val) for val in line_values[-4:]])
#             if len(values) == 4:
#                 prediction.append(values)
#                 values = []

#     # bbox = privacy_predict_info[str(i)]['bbox'][:box_topk]
#     bbox = privacy_predict_info.get(str(i))
#     if bbox is not None:
#         bbox = bbox['bbox'][:box_topk]
#     else:
#         continue  # Skip this iteration if the key is not found

#     box_labels = privacy_predict_info[str(i)]['bbox_labels'][:box_topk]

#     # # 绘制图片和边界框
#     # plt.figure()
#     # plt.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))

#     for pred in prediction:
#         x, y, w, h = pred
#         x1 = x * width
#         y1 = y * height
#         x2 = x1 + w * width
#         y2 = y1 + h * height
#         my_box = [x1, y1, x2, y2]

#         # 计算与my_box最接近的边界框
#         min_distance = math.inf
#         nearest_box_index = None

#         for i, box in enumerate(bbox):
#             bx1, by1, bx2, by2 = box
#             box_center_x = (bx1 + bx2) / 2
#             box_center_y = (by1 + by2) / 2
#             distance = math.sqrt((box_center_x - (my_box[0] + my_box[2]) / 2)**2 +
#                                 (box_center_y - (my_box[1] + my_box[3]) / 2)**2)

#             if distance < min_distance:
#                 min_distance = distance
#                 nearest_box_index = i

#         # 输出最接近的边界框索引
#         nearest_box_number = nearest_box_index
#         # print("最接近的边界框是第", nearest_box_number, "个     ", privacy_obj_class[box_labels[nearest_box_index]])
#         true_label[nearest_box_index] = 1
#         # # 绘制my_box
#         # rect = plt.Rectangle((my_box[0], my_box[1]), my_box[2] - my_box[0], my_box[3] - my_box[1],
#         #                     fill=False, edgecolor='b', linewidth=2)
#         # plt.gca().add_patch(rect)

#         # # 绘制最接近的边界框
#         # x1, y1, x2, y2 = bbox[nearest_box_index]
#         # rect = plt.Rectangle((int(x1), int(y1)), int(x2) - int(x1), int(y2) - int(y1), fill=False, edgecolor='r', linewidth=2)
#         # plt.gca().add_patch(rect)
    

#     # # 显示图像和边界框
#     # plt.axis('off')
#     # plt.show()
#     all_true_label.append(true_label)

# print(f"Saving at {save_dir}......................")
# privacy_imgs_info["all_true_label"] = all_true_label
# # 将更新后的 JSON 数据保存回文件
# with open(f'{save_dir}/reltr_custom_data_info.json', 'w') as f:
#     json.dump(privacy_imgs_info, f)













