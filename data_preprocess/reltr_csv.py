import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import gc
from PIL import Image
import os
import requests
import matplotlib.pyplot as plt
import csv

from main import parse_arguments
from Reltr.backbone import Backbone, Joiner
from Reltr.position_encoding import PositionEmbeddingSine
from Reltr.transformer import Transformer
from Reltr.reltr import RelTR

"""
第一步：场景图检测，下一步：place_csv
"""

torch.cuda.empty_cache()

#目标与关系列表
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

args = parse_arguments()
# 设置设备为CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.1, nhead=8,
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)

model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
              num_entities=100, num_triplets=200)

# The checkpoint is pretrained on Visual Genome
# ckpt = torch.hub.load_state_dict_from_url(
#     url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
#     map_location='cpu', check_hash=True)
ckpt = torch.load('../ckpt/checkpoint0149.pth', map_location=device)
model.load_state_dict(ckpt['model'])
model.to(device)
model.eval()

# Some transformation functions
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    device = out_bbox.device  # 获取设备
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32,device=device)
    return b

# 可更换图片
#url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Yellow_Happy.jpg/1200px-Yellow_Happy.jpg'
def read_img(folder_path):
    # 确保输入的路径是存在的
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"地址 {folder_path} 不存在.")

    # 获取文件夹内所有文件的名称
    files = os.listdir(folder_path)

    # 初始化空列表来存储图片对象
    images = []

    # 遍历文件夹内的所有文件
    for file in files:
        # 检查文件是否是jpg格式
        if file.lower().endswith('.jpg'):
            # 获取不包含扩展名的文件名
            base_filename = os.path.splitext(file)[0]
            # 构造文件的完整路径
            file_path = os.path.join(folder_path, file)

            # 使用Image.open打开图片，并将其添加到列表中
            try:
                im = Image.open(file_path)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                print(f"Image Name: {base_filename}, Size: {im.size}, Mode: {im.mode}")
                images.append((im, base_filename))
            except IOError:
                print(f"Could not open {file_path}")

    return images

#将处理后的内容记录到表格中
def inference_save(images, topk, csv_path):
  for im, im_name in images:
    # 该图片csv存放地址
    csv_file_path = os.path.join(csv_path, f'{im_name}.csv')

    # 检查是否已经存在同名的csv文件，如果存在则跳过该图片
    if os.path.exists(csv_file_path):
        print(f"CSV file {csv_file_path} already exists. Skipping...")
        continue
    else:
        print(f"CSV file {csv_file_path} is created.")

    #transformer编码
    img = transform(im).unsqueeze(0).to(device)  # 确保输入张量在CPU上

    # 在模型中传播
    try:
        outputs = model(img)
    except Exception as e:
        print(f"生成场景图 {csv_file_path} 时出错: {e}")
        continue

    # 只保留置信度大于0.3的预测
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    #返回张量元素的逻辑and结果，筛选出满足三个条件（对象和关系）同时成立的索引
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,probas_obj.max(-1).values > 0.3))

    # 如果没有满足条件的预测，创建一个空的CSV文件，并写入预设的第一行
    if keep.sum() == 0:
        print(f"No valid predictions for image {im_name}. Saving empty CSV file...")
        csv_file_path = os.path.join(csv_path, f'{im_name}.csv')
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['sub', 'rel', 'onj', 'sxmin', 'symin', 'sxmax', 'symax', 'oxmin', 'oymin', 'oxmax', 'oymax'])
        continue

    # convert boxes from [0; 1] to image scales从[0;1]图像尺度
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    # 先以元组形式返回所有预测的三元组，然后取置信度最大的topk个三元组返回
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]

    # save the attention weights保存注意力权重
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )]
    try:
        with torch.no_grad():
            # propagate through the model(不能少这一行)
            outputs = model(img)

            for hook in hooks:
                hook.remove()

            # don't need the list anymore
            conv_features = conv_features[0]
            dec_attn_weights_sub = dec_attn_weights_sub[0]
            dec_attn_weights_obj = dec_attn_weights_obj[0]

            # get the feature map shape im是加载的图片
            h, w = conv_features['0'].tensors.shape[-2:]
            im_w, im_h = im.size

            #单个图片建立空列表
            csv_data = []
            #读取索引，目标框
            for idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                    zip(keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                one_sub = CLASSES[probas_sub[idx].argmax()]
                one_rel = REL_CLASSES[probas[idx].argmax()]
                one_obj = CLASSES[probas_obj[idx].argmax()]

                row_data = [one_sub, one_rel, one_obj, sxmin, symin, sxmax, symax, oxmin, oymin, oxmax, oymax]
                csv_data.append(row_data)

            with open(csv_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['sub','rel','onj','sxmin','symin','sxmax','symax','oxmin','oymin','oxmax','oymax'])
                for r_data in csv_data:
                   writer.writerow(r_data)
            print(f"已储存csv文件至 {csv_file_path}")
    except Exception as e:
        print(f"生成场景图 {csv_file_path} 时出错: {e}")
        continue
    # 释放不再需要的变量
    del img, outputs, probas, probas_sub, probas_obj, keep, sub_bboxes_scaled, obj_bboxes_scaled, keep_queries, indices
    del conv_features, dec_attn_weights_sub, dec_attn_weights_obj, csv_data
    gc.collect()


def main(args):
    # 读取图片数据文件路径
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    data_file_path = os.path.join(dataset_dir, f"images/{args.data_type}")
    images = read_img(data_file_path)
    print(len(images))
    # for image, name in images:
    #     print(f"Image Name: {name}, Size: {image.size}, Mode: {image.mode}")

    plt.imshow(images[4][0])
    #plt.show()
    #输入：图片集合，topk三元组， csv存放地址
    csv_file_path = os.path.join(dataset_dir, f"SC/{args.data_type}")
    os.makedirs(csv_file_path, exist_ok=True)
    inference_save(images, 20, csv_file_path)

if __name__ == '__main__':
    main(args)