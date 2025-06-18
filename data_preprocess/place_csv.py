import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
import gc
from ckpt import wideresnet
from nudenet import NudeDetector
from main import parse_arguments
import matplotlib.pyplot as plt
import requests
import csv

"""
第二步：场景检测和裸露检测,下一步：yolo检测
"""

def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels():
    # prepare all the labels
    # scene category relevant
    file_name_category = '../ckpt/categories_places365.txt'
    if not os.access(file_name_category, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = '../ckpt/IO_places365.txt'
    if not os.access(file_name_IO, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
        os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # # scene attribute relevant
    # file_name_attribute = 'labels_sunattribute.txt'
    # if not os.access(file_name_attribute, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
    #     os.system('wget ' + synset_url)
    # with open(file_name_attribute) as f:
    #     lines = f.readlines()
    #     labels_attribute = [item.rstrip() for item in lines]
    # file_name_W = 'W_sceneattribute_wideresnet18.npy'
    # if not os.access(file_name_W, os.W_OK):
    #     synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
    #     os.system('wget ' + synset_url)
    # W_attribute = np.load(file_name_W)

    # return classes, labels_IO, labels_attribute, W_attribute
    return classes, labels_IO

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14

    # model_file = 'wideresnet18_places365.pth.tar'
    model_file = '../ckpt/wideresnet18_places365.pth.tar'
    # os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

    model.eval()
    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model

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
                images.append((im, base_filename))
            except IOError:
                print(f"Could not open {file_path}")

    return images

def inference_save(images, topk, csv_path):
  for im, im_name in images:
    # 该图片csv存放地址
    csv_file_path = os.path.join(csv_path, f'{im_name}.csv')

    # 检查CSV文件是否存在以及是否包含“environment”字段
    skip_inference = False
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if 'environment' in row:
                    print(f"跳过推理，CSV文件已包含环境信息：{csv_file_path}")
                    skip_inference = True
                    break
    else:
        skip_inference = True
    if skip_inference:
        continue

    #transformer编码
    input_img = V(tf(im).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    one_data = []
    # 假如室内室外识别
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        one_data.append('indoor')
    else:
        one_data.append('outdoor')

    # 加入topk场景识别
    for i in range(0, topk):
      one_data.append(classes[idx[i]])

    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['environment','scene'])
        writer.writerow(one_data)
    print(f"已储存csv文件至 {csv_file_path}")

    # 释放不再需要的变量
    del input_img, logit, h_x, probs, idx, one_data
    gc.collect()

nude_detector = NudeDetector()
def nude_save(folder_path, csv_path):
  # 确保输入的路径是存在的
  if not os.path.exists(folder_path):
      raise FileNotFoundError(f"地址 {folder_path} 不存在.")

  # 获取文件夹内所有文件的名称
  files = os.listdir(folder_path)

  # 遍历文件夹内的所有文件
  for file in files:
      #存储csv用空列表
      one_data = []
      # 检查文件是否是jpg格式
      if file.lower().endswith('.jpg'):
          # 获取不包含扩展名的文件名
          base_filename = os.path.splitext(file)[0]
          # 构造文件的完整路径
          file_path = os.path.join(folder_path, file)

          # 该图片csv存放地址
          csv_file_path = os.path.join(csv_path, f'{base_filename}.csv')
          # 检查CSV文件是否存在以及是否包含“environment”字段
          skip_inference = False
          if os.path.exists(csv_file_path):
              with open(csv_file_path, 'r') as csvfile:
                  reader = csv.reader(csvfile)
                  for row in reader:
                      if 'nude_class' in row:
                          print(f"跳过推理，CSV文件已包含裸露检测信息：{csv_file_path}")
                          skip_inference = True
                          break
          else:
              skip_inference = True
          if skip_inference:
              continue

          try:
              # 返回检测结果
              nude_class = nude_detector.detect(file_path)
          except Exception as e:
              print(f"检测文件 {file_path} 时出错: {e}")
              # nude_class = []  # 如果发生错误，分配一个空列表
              continue

          for nude in nude_class:
              row_nude = []
              row_nude.append(nude['class'])
              row_nude.append(nude['score'])
              row_nude.append(nude['box'])
              one_data.append(row_nude)

          try:
              with open(csv_file_path, 'a', newline='') as csvfile:
                  writer = csv.writer(csvfile)
                  writer.writerow(['nude_class','score','box'])
                  for n_data in one_data:
                    writer.writerow(n_data)
              print(f"(nude)已储存csv文件至 {csv_file_path}")
          except IOError:
              print(f"Could not open {file_path}")


# load the labels
classes, labels_IO = load_labels()
print(labels_IO)

# load the model
features_blobs = []
model = load_model()

# load the transformer
tf = returnTF() # image transformer

# get the softmax weight
params = list(model.parameters())
weight_softmax = params[-2].data.numpy()
weight_softmax[weight_softmax<0] = 0


def main(args):
    # 图片数据文件路径示例
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    data_file_path = os.path.join(dataset_dir, f"images/{args.data_type}")
    images = read_img(data_file_path)
    print(len(images))
    for image, name in images:
        print(f"Image Name: {name}, Size: {image.size}, Mode: {image.mode}")

    plt.imshow(images[4][0])
    # plt.show()

    #图片集合，场景类别输出topk， csv存放地址
    csv_file_path = os.path.join(dataset_dir, f"SC/{args.data_type}")
    inference_save(images, 5, csv_file_path)
    #图片地址，csv存放地址,输出裸露检测结果
    nude_save(data_file_path, csv_file_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)