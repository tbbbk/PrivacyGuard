from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import torch


def calculate_index(pred, y):
    pred_np = pred.numpy()
    y_np = y.numpy()

    # Calculate True Positives, True Negatives, False Positives, False Negatives
    TP = ((pred_np == 1) & (y_np == 1)).sum()
    TN = ((pred_np == 0) & (y_np == 0)).sum()
    FP = ((pred_np == 1) & (y_np == 0)).sum()
    FN = ((pred_np == 0) & (y_np == 1)).sum()
    confusion_matrix_str = (
    f"{'':^10}{'预测为正类':^12}{'预测为负类':^12}{'总计':^7}\n"
    f"{'实际为正类':^10}{TP:>12}{FP:>12}{TP+FP:>7}\n"
    f"{'实际为负类':^10}{FN:>12}{TN:>12}{FN+TN:>7}\n"
    f"{'总计':^10}{TP+FN:>12}{FP+TN:>12}{TP+FN+FP+TN:>7}"
)

    # 打印混淆矩阵
    print(confusion_matrix_str)
    
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
    return accuracy, precision, recall, f1_score
    
    
def ROC_PR(pred, y):
    fpr, tpr, thresholds_roc = roc_curve(y.cpu().numpy(), pred.cpu().numpy())
    roc_auc = auc(fpr, tpr)

    # 计算精确率（Precision），召回率（Recall）以及阈值
    precision, recall, thresholds_pr = precision_recall_curve(y.cpu().numpy(), pred.cpu().numpy())

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('/home/zhuohang/SGG-GCN/images/{}.jpg'.format('ROC'))
    plt.show()

    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig('/home/zhuohang/SGG-GCN/images/{}.jpg'.format('PR'))
    plt.show()

