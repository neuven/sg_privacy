import pandas as pd
import requests
import os
from tqdm import tqdm

# 定义文件路径
# csv_file_path = 'F:/data/privacy_detection_dataset_v2/ImFiles/test_urls.csv'  # 替换为你的 CSV 文件路径
# download_folder = 'F:/data/privacy_detection_dataset_v2/images/test'  # 替换为你要保存图片的文件夹路径
csv_file_path = '/home/liangxy/pycharm/sg_privacy/data/picalert/url.csv'  # 替换为你的 CSV 文件路径
download_folder = '/home/liangxy/pycharm/sg_privacy/data/picalert/images'  # 替换为你要保存图片的文件夹路径
dataset = "picalert"
# 创建文件夹如果不存在
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# 读取 CSV 文件
df = pd.read_csv(csv_file_path)

# 遍历每一行的图片地址并下载
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Downloading images"):
    if index <28000:
        continue

    if dataset == "picalert":
        img_url = row[1]
        photo_id = row[0]
        img_name = f"{photo_id}.jpg"
    else:
        # 获取图片地址
        img_url = row[0]
        # 获取图片名称
        # 提取photo_id并构建新的图片名称
        photo_id = img_url.split('/')[-1].split('_')[0]
        img_name = f"{photo_id}.jpg"
    if not isinstance(img_url, str):
        continue

    img_path = os.path.join(download_folder, img_name)
    # 检查文件是否已经存在
    if os.path.exists(img_path):
        print(f"File already exists: {img_name}, skipping download.")
        continue
    else:
        print(f"Downloaded: {img_url}")

    # 下载图片
    img_data = requests.get(img_url).content


    with open(img_path, 'wb') as handler:
        handler.write(img_data)


print("All images have been downloaded.")


