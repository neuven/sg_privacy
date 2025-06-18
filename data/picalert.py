import os
import pandas as pd
import requests
import flickrapi
from flickrapi.exceptions import FlickrError
from tqdm import tqdm
import time

# Flickr API 配置
api_key = '9dec414436d58bedcfeadda8eb59250a'
api_secret = '0d99fb93a3c2cfdd'

# 初始化 FlickrAPI
flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json',cache=True)
# os.environ['NO_PROXY'] = 'api.flickr.com'
# 设置 HTTP 和 HTTPS 代理
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'


def download_image(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"Image saved to {save_path}")
    else:
        print(f"Failed to download image from {url}")


def get_flickr_image_url(photo_id):
    try:
        # 获取照片信息
        photo_info = flickr.photos.getSizes(photo_id=photo_id)
        # 获取原始图像 URL
        for size in photo_info['sizes']['size']:
            if size['label'] == 'Original':
                return size['source']
        # 如果没有原始图像，则获取最大尺寸的图像
        return photo_info['sizes']['size'][-1]['source']
    except FlickrError as e:
        print(f"Error retrieving Flickr image {photo_id}: {e}")
        return None



def save_flickr_image_urls(csv_file, save_csv):
    # 读取 CSV 文件
    df = pd.read_csv(csv_file)

    # 如果保存的 CSV 文件存在，读取其中已有的数据
    if os.path.exists(save_csv):
        existing_df = pd.read_csv(save_csv)
        existing_ids = list(existing_df['photo_id'])
    else:
        existing_df = pd.DataFrame(columns=['photo_id', 'image_url'])
        existing_ids = []
        existing_df.to_csv(save_csv, index=False)  # 初始化保存文件

    # 创建一个空的列表用于保存图片URL和photo_id
    data = []

    # 遍历每一行，获取对应的 Flickr 图片 URL
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing photos"):
        photo_id = str(row[0])  # 假设图片编号在第一列
        if int(photo_id) in existing_ids:
            # 如果 photo_id 已存在，跳过处理
            print(f"Photo ID {photo_id} already exists. Skipping.")
            continue
        # try:
        # 增加请求间隔，避免触发服务器限制
        # time.sleep(1)  # 每次请求间隔1秒
        image_url = get_flickr_image_url(photo_id)
        # 将 photo_id 和 image_url 添加到新的 DataFrame
        new_data = pd.DataFrame({'photo_id': [photo_id], 'image_url': [image_url]})
        # 将新数据追加到文件中
        with open(save_csv, 'a', newline='') as f:
            new_data.to_csv(f, header=f.tell() == 0, index=False)
        print(f"Image URL for {photo_id} saved.")
        # print(f"URL for image {photo_id} retrieved: {image_url}")
        # except Exception as e:
        #     print(f"Failed to retrieve image {photo_id}: {e}")


try:
    response = requests.get("https://api.flickr.com/services/rest/")
    print("Status Code:", response.status_code)
except Exception as e:
    print("Request failed:", e)

# 示例使用
csv_file = 'F:/data/picalert/db.csv'
save_csv = 'F:/data/picalert/url.csv'
save_dir = 'F:/data/picalert/images'

save_flickr_image_urls(csv_file, save_csv)
# download_flickr_images_from_csv(csv_file, save_dir)



