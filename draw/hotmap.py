import torch
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
class MyGCNModel(torch.nn.Module):
    def __init__(self):
        super(MyGCNModel, self).__init__()
        # 定义图卷积层
        self.conv1 = GCNConv(10, 20)
        self.conv2 = GCNConv(20, 30)
        self.conv3 = GCNConv(30, 40)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        
        return x

# 初始化模型
model = MyGCNModel()

# 定义一个字典来存储每一层的输出
layer_outputs = {}

# 定义一个钩子函数
def get_layer_output(module, input, output):
    layer_outputs[str(module)] = output

# 注册钩子
for layer in model.children():
    layer.register_forward_hook(get_layer_output)

# 假设的图数据输入
num_nodes = 5
num_node_features = 10
x = torch.randn(num_nodes, num_node_features)  # 节点特征sudo
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)  # 边

# 创建一个图数据对象
data = Data(x=x, edge_index=edge_index)

# 计算模型输出（在这个过程中，钩子函数将会被调用）
model.eval()
with torch.no_grad():
    model(data)




# 绘制每一层的输出热力图，每个热力图包含所有节点的输出
for layer, output in tqdm(layer_outputs.items()):
    output = output.detach().numpy()  # 将输出转换为numpy数组
    # output= np.mean(output, axis=1, keepdims=True)
    #output= np.sum(output, axis=1, keepdims=True)
    #output= np.max(output, axis=1, keepdims=True)
    #output= np.min(output, axis=1, keepdims=True)
    output= np.median(output, axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    # 直接绘制热力图，无需reshape。每行代表一个节点的输出特征。
    ax = sns.heatmap(output, annot=True, fmt="f", cmap='viridis')
    plt.title(f"Heatmap for {layer}")
    
    # 构建图层名字的字符串表示，用于文件名
    layer_name = str(layer).replace('\n', '').replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
    
    # 保存热力图到指定路径
    img_path = f"/home/zhuohang/SGG-GCN/draw/img/{layer_name}.jpg"
    plt.savefig(img_path)
    plt.close()  # 关闭图表以释放内存
