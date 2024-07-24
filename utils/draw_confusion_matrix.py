import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools

def generate_confusion_matrix(pred, y, classes=['Public', 'Private'], file_name="confusion_matrix"):
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


if __name__ == '__main__':
    '''
    绘制confusion matrix部分
    '''
    generate_confusion_matrix(dataloader_path='/home/bingkui/workspace/MIL/Dataloader/Hockey/RandomcropHockeyTestGraphDataloader.pth',
                              model_path='/home/bingkui/workspace/MIL/ckpt/HockeyRandomCrop/model_Hockey_18_350.pth', 
                              classes=['none_violence', 'violence'],
                              filename='HockeyRandomCrop')