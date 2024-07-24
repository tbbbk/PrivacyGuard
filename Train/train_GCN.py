import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, Subset
import sys
import yaml
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import os
with open("./configs/model.yaml", "r") as f:
    model_yaml = yaml.safe_load(f)

project_dir = model_yaml["project_dir"]
sys.path.append(project_dir)

import utils.load_data as ld
from Model.model import GCN, privacy_dataset, EarlyStopping, GAT


def generate_confusion_matrix(pred, y, classes=['Private', 'Public'], file_name="confusion_matrix"):
    matrix = [[0, 0], [0, 0]]
    assert len(pred) == len(y)
    for i in range(len(pred)):
        if pred[i] == y[i]:
            if pred[i] == 1:
                matrix[0][0] = matrix[0][0] + 1
            elif pred[i] == 0:
                matrix[1][1] = matrix[1][1] + 1
        elif pred[i] != y[i]:
            if pred[i] == 1:
                matrix[1][0] = matrix[1][0] + 1
            elif pred[i] == 0:
                matrix[0][1] = matrix[0][1] + 1
    plot_confusion_matrix(matrix, classes=classes, filename=file_name, normalize=True, title='Normalized confusion matrix')
    return matrix
    
            
def plot_confusion_matrix(cm, classes, filename, normalize=False, title='Confusion matrix', cmap = plt.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = np.array(cm)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('/home/zhuohang/SGG-GCN/images/{}.jpg'.format(filename))
    print('Matrix save to /home/zhuohang/SGG-GCN/images/{}.jpg'.format(filename))

def calculate_index(pred, y):
    pred_np = pred.numpy()
    y_np = y.numpy()

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = ((pred_np == 1) & (y_np == 1)).sum()
    TN = ((pred_np == 0) & (y_np == 0)).sum()
    FP = ((pred_np == 1) & (y_np == 0)).sum()
    FN = ((pred_np == 0) & (y_np == 1)).sum()

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Precision
    precision = TP / (TP + FP)

    # Recall
    recall = TP / (TP + FN)

    # F1 Score
    f1_score = 2 * precision * recall / (precision + recall)
    
    # Specificity
    specificity = TN / (TN + FP)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_score)
    print("Specificity:", specificity)
    
    
# 创建数据加载器
def create_data_loader(data_list, batch_size, shuffle=True):
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    for batch in tqdm(data_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, batch.y)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader.dataset)

# 测试函数
def test(model, data_loader, device):
    pred_node, y_node, ans_origin_value = [], [], []
    model.eval()
    total_len = 0
    correct = 0
    for idx,batch in tqdm(enumerate(data_loader)):
        batch = batch.to(device)
        output = model(batch)
        pred = output.argmax(dim=1)
        correct += pred.eq(batch.y).sum().item()
        pred_node.append(output.argmax(dim=1).tolist())
        y_node.append(batch.y.tolist())
        total_len += batch.y.shape[0]
    # print(total_len)
    return correct / total_len, pred_node, y_node


def my_print(print_thing, s):
    s += str(print_thing) + "/n"
    print(print_thing)
    return s
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="mosaic",help='the dataset want to change(privacy_dataset_1000, mosaic, privacy_dataset_1000_privacy, privacy_dataset_1000_public)')
    parser.add_argument('--embed_mode', default="nn_embed", help="nn_embed or onehot")
    parser.add_argument('--top_n', default=30, type=int , help="int num(0-80)")
    parser.add_argument("--least_trust", default=0.5,type=float, help="0-1")
    parser.add_argument("--relation_node", default="add", help="add or multi")
    parser.add_argument("--epoch_num", default=200, type=int , help="int epoch num")
    parser.add_argument("--train_ratio", default=0.8, type= float, help="0-1")
    parser.add_argument("--lr", default=0.01, type=float, help="0-1")
    parser.add_argument("--save_num", default=10, type=int, help="How many intervals to save(1-inf)")
    parser.add_argument("--batch_size", default=100, type=int, help="batch_size")

    training_time = time.strftime('%Y-%m-%d-%H-%M-%S')
    save_folder_path = f"/home/zhuohang/SGG-GCN/pth/gcn_pth/{training_time}"

    args = parser.parse_args()
    dataset_name = args.dataset
    embed_mode = args.embed_mode
    top_n = args.top_n
    least_trust = args.least_trust
    relation_node = args.relation_node
    epoch_num = args.epoch_num
    model_train_ratio = args.train_ratio
    model_lr = args.lr
    save_num = args.save_num
    model_batch_size= args.batch_size
    txt_output = ""
    txt_output=my_print("-"*20,txt_output)
    txt_output=my_print(f"training_time:{training_time}",txt_output)
    txt_output=my_print("-"*20,txt_output)
    txt_output=my_print("training...........",txt_output)
    txt_output=my_print("-"*20,txt_output)
    txt_output=my_print("loading args from terminal........ \n get args:",txt_output)
    txt_output=my_print(f"dataset_name:{dataset_name}",txt_output)
    txt_output=my_print(f"embed_mode:{embed_mode}",txt_output)
    txt_output=my_print(f"top_n:{top_n}",txt_output)
    txt_output=my_print(f"least_trust:{least_trust}",txt_output)
    txt_output=my_print(f"relation_node:{relation_node}",txt_output)
    txt_output=my_print(f"epoch_num:{epoch_num}",txt_output)
    txt_output=my_print(f"train_ratio:{model_train_ratio}",txt_output)
    txt_output=my_print(f"lr:{model_lr}",txt_output)
    txt_output=my_print(f"save_num:{save_num}",txt_output)
    txt_output=my_print(f"batch_size:{model_batch_size}",txt_output)
    txt_output=my_print("-"*20,txt_output)
    # 示例数据
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
    with_link = model_info["model"]["with_link"]
    txt_output=my_print(f"if train with link embeding:{with_link}", txt_output)
    txt_output=my_print("-"*20,txt_output)
    txt_output=my_print("loading args from /configs/model.yaml........ \n get args:",txt_output)
    txt_output=my_print("input param:",txt_output)
    if(embed_mode == "nn_embed",txt_output):
        txt_output=my_print("nn_embed:",txt_output)
        txt_output=my_print(f"node_features_dim:{node_features_dim}",txt_output)
        txt_output=my_print(f"edge_embed_dim:{edge_embed_dim}",txt_output)
    elif(embed_mode == "onehot",txt_output):
        txt_output=my_print("onehot:",txt_output)
        txt_output=my_print(f"node_features_dim == node_num_categories:{node_num_categories}",txt_output)
        txt_output=my_print(f"edge_embed_dim == edge_num_relations:{edge_num_relations}",txt_output)
    txt_output=my_print("model param:",txt_output)
    txt_output=my_print(f"input_layer_dim:{input_layer_dim}",txt_output)
    txt_output=my_print(f"hidden_layer_dim:{hidden_layer_dim}",txt_output)
    txt_output=my_print(f"output_layer_dim:{output_layer_dim}",txt_output)
    txt_output=my_print("-"*20,txt_output)
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    txt_output=my_print(f"device:{device}",txt_output)
    txt_output=my_print("-"*20,txt_output)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    txt_output=my_print(f"pth_save_dir:{save_folder_path}", txt_output)
    
    # 创建完整数据集
    full_dataset = privacy_dataset(dataset_name)
    # 创建数据加载器
    data_loader = create_data_loader(full_dataset, batch_size=model_batch_size, shuffle=True)

    # 创建模型
    model = GCN(in_channels=input_layer_dim, hidden_channels=hidden_layer_dim, out_channels=output_layer_dim).to(device)
    model.load_state_dict(torch.load('/home/zhuohang/SGG-GCN/pth/gcn_pth/2024-03-05-12-40-32/GCN_model_200.pth'))
    # 定义损失函数和优化器
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_lr)

    # 划分训练集和测试集
    train_ratio = model_train_ratio  # 训练集比例
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    test_acc, pred_node, y_node = test(model, test_dataset, device)
    pred = torch.cat([torch.tensor(sublist) for sublist in pred_node], dim = 0)
    y = torch.cat([torch.tensor(sublist) for sublist in y_node], dim = 0)
    confusion_matrix = generate_confusion_matrix(pred, y)
    calculate_index(pred=pred, y=y)
    
    # # 训练循环
    # for epoch in range(epoch_num):
    #     txt_output=my_print(f"Training............. Round {epoch + 1} ............................", txt_output)
    #     train_loss = train(model, train_dataset, optimizer, criterion, device)
    #     test_acc, _, _ = test(model, test_dataset, device)
    #     txt_output=my_print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Test Accuracy: {test_acc}',txt_output)
        
    #     # 保存模型
    #     if (epoch + 1) % save_num == 0:
    #         model_path = f"{save_folder_path}/GCN_model_{epoch+1}.pth"
    #         torch.save(model.state_dict(), model_path)
    #         txt_output=my_print(f"Model saved at {model_path}", txt_output)

    # with open(f"{save_folder_path}/model.txt", "w")as f:
    #     f.write(txt_output)
