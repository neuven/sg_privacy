import torch
import random
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn as nn


def calculate_iou(box1, box2):
    """计算两个框的交并比（IoU）"""
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    # 计算交集的坐标
    xi1, yi1 = max(x1, x1_), max(y1, y1_)
    xi2, yi2 = min(x2, x2_), min(y2, y2_)

    # 计算交集面积
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # 计算两个框的面积
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    # 计算并集面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    if union_area == 0:
        return 0.0  # 如果并集面积为零，返回IoU为0
    iou = inter_area / union_area
    return iou

def get_font_size(text, max_width, initial_font_size=12):
    # 递减字体大小直到文本宽度适应最大宽度
    font_size = initial_font_size
    while len(text) * font_size > max_width and font_size > 5:
        font_size -= 1
    return font_size

def load_data_from_npy(file_path):
    return np.load(file_path, allow_pickle=True)


def save_model(model, name):
    torch.save(model, name)


def load_model(name):
    model = torch.load(name)
    return model


def test_model(model, data_loader, device):
    model.eval()  # 将模型设置为评估模式
    criterion = nn.CrossEntropyLoss()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    # with torch.no_grad():  # 由于有register，貌似不能不用梯度了。
    for features, adjacency_matrix, labels, subgraph_indices_list, padded_extra_matrices in data_loader:
        features, adjacency_matrix, labels, extra_matrix = features.to(device), adjacency_matrix.to(device), labels.to(device), padded_extra_matrices.to(device)

        # 前向传播
        outputs, attn_weights = model(features, adjacency_matrix, subgraph_indices_list, extra_matrix)

        # 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # 预测类别
        _, predicted = torch.max(outputs.data, 1)

        # 收集所有的真实标签和预测标签
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    # 输出真实标签和预测标签
    # for true_label, predicted_label in zip(all_labels, all_predictions):
    #     print(f"True Label: {true_label}, Predicted Label: {predicted_label}")

    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    # 使用get_metrics函数计算评估指标
    acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1 = get_metrics(all_labels, all_predictions)

    # 输出评估指标
    print(f"Accuracy: {acc:.2f}%")
    print(f"Public Precision: {pub_prec:.4f}")
    print(f"Public Recall: {pub_rec:.4f}")
    print(f"Private Precision: {priv_prec:.4f}")
    print(f"Private Recall: {priv_rec:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    print("=" * 50)

    return avg_loss, acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1


# def save_checkpoint(classifier, optimizer, epoch):
#     checkpoint_state = {
#         "classifier": classifier.state_dict(),
#         "optimizer": optimizer.state_dict(),
#         "epoch": epoch
#     }
#
#     checkpoint_path = ("./checkpoints/model.ckpt-{}_" + model_name + ".pt").format(epoch)
#     torch.save(checkpoint_state, checkpoint_path)
#     print("Saved checkpoint: {}".format(checkpoint_path))
#
#     return checkpoint_path

def print_loader(data_loader):
    for combined_features, adjacency_matrix, label, subgraph_indices_list, extra_matrix in data_loader:
        # 此处可以将数据输入到模型中进行训练
        print("特征：", combined_features.shape, combined_features)
        print("邻接矩阵：", adjacency_matrix.shape, adjacency_matrix)
        print("标签：", label)
        print("子图索引：", subgraph_indices_list)
        print("图像特征矩阵：", extra_matrix.shape, extra_matrix)
        break  # 仅打印第一个batch的数据


def set_seed(seed_val):
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_metrics(truths, preds):
    cm = confusion_matrix(truths, preds)
    tpos, fneg, fpos, tneg = cm.ravel()
    acc = 100 * (float(tneg) + float(tpos)) / float(tneg + tpos + fneg + fpos)
    pub_prec = tpos / float(tpos + fpos)  # 计算公开类的精确率
    pub_rec = tpos / float(tpos + fneg)   # 计算公开类的召回率
    priv_prec = tneg / float(tneg + fneg)  # 计算私有类的精确率
    priv_rec = tneg / float(tneg + fpos)  # 计算私有类的召回率
    f1_priv = (2*priv_rec*priv_prec)/(priv_prec+priv_rec)  # 计算私有类的F1分数
    f1_pub = (2*pub_prec*pub_rec)/(pub_prec+pub_rec)  # 计算公开类的F1分数
    macro_f1 = (f1_priv+f1_pub)/2  # 计算宏观平均F1分数

    return acc, pub_prec, pub_rec, priv_prec, priv_rec, cm, macro_f1


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Code originally from https://pjreddie.com/darknet/yolo/.
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5 : 5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = (
                max_detections if output[image_i] is None else torch.cat((output[image_i], max_detections))
            )

    return output


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    Code originally from https://pjreddie.com/darknet/yolo/.
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def load_classes(path):
    """
    Loads class labels at 'path'
    Code originally from https://pjreddie.com/darknet/yolo/.
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def parse_model_config(path):
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['): # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parses the data configuration file
    Code originally from https://pjreddie.com/darknet/yolo/"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options