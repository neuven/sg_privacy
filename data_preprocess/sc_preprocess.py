"""
第四步：将检测的csv表格内的信息整理成需要的信息，全部存到json文件中。
罗列需要的信息：
1.节点特征（应该需要对应的名称）
根据场景图生成+目标检测出来的多个目标，比如：人，椅子
需要识别哪些节点是同一个，通过位置来判断。
特殊检测：裸体，牌照，文件，数量（数字），场景。补充目标：使用yolo检测的目标，在场景图内容不够时作为添加。
2.关系特征（应该需要对应的名称）
根据场景图生成+规则，将节点连接起来的关系，比如：旁边
规则：A数量有B个；A场景中有B（裸体/牌照/文件/补充目标）。
3.邻接矩阵
根据场景图生成+规则，节点和关系的连接关系。
4.子图索引
场景图生成：每个三元组，限定数量；规则：自定义。
5.标签
隐私/公开
6.每个节点的图像特征
先将每个节点的位置信息提取出来。场景信息的图像特征为全图。
"""
import os
import pandas as pd
from tqdm import tqdm
import cv2
import json
from main import parse_arguments
import re
import csv


# 比较两个节点的位置信息
def is_same_node(pos1, pos2, tolerance):
    pos1_numbers = [float(p) for p in pos1]
    pos2_numbers = [float(p) for p in pos2]
    for a, b in zip(pos1_numbers, pos2_numbers):
        if abs(a - b) >= tolerance:
            return False
    return True

def match_tensor(content):
    match = re.search(r'tensor\((-?[\d.]+)', content)
    return match.group(1)

def process_nude_position(pos_str):
    # 去除两端的方括号并将字符串按逗号分割，然后去除多余的空格
    f_process = [s.strip() for s in pos_str.strip('[]').split(',')]
    # 将字符串转换为浮点数进行计算
    x, y, w, h = map(float, f_process)
    # 计算新的坐标
    x2, y2 = x + w, y + h
    # 将结果转换回字符串形式
    final_process = [str(int(x)), str(int(y)), str(int(x2)), str(int(y2))]
    return final_process


def process_csv(csv_file_path: str, max_rows: int = 20, tolerance: float = 20.0):
    # 读取表格数据
    # df = pd.read_excel('your_file.xlsx')  # 假设表格是Excel格式
    # 读取csv文件时使用
    df = pd.read_csv(csv_file_path)

    # 提取max_rows组场景图三元组内容
    sc_data = []
    for idx, row in df.iloc[0:].iterrows():
        if idx >= max_rows or row['sub'] == 'environment':
            break
        sc_data.append(row)
    sc_data = pd.DataFrame(sc_data)


    # 查找不同的行索引
    environment_index = None
    nude_class_index = None
    yolo_number_index = None
    yolo_index = None
    license_index = None
    node_list = []
    relationship_list = []
    yolo_name = []
    yolo_position = []
    nude_class = []
    nude_position = []
    license_name = []
    license_position = []
    yolo_count_dict = {}

    for idx, row in df.iterrows():
        if row['sub'] == 'environment':
            # 将场景名称存在节点的第一个
            node_list.append(df.iloc[idx + 1, 1])
        if row['sub'] == 'nude_class':
            nude_class_index = idx
        # 这个地方要修改下表格，名字要不一样比较好。另外，在生成表格时就改下yolo的节点名字吧，让它和场景图一样。
        if row['sub'] == 'yolo_count':
            yolo_number_index = idx
        if row['sub'] == 'yolo_object':
            yolo_index = idx
        if row['sub'] == 'l_object':
            license_index = idx
            break

    # 确保找到的索引有效且 "nude_class" 在 "object" 之前
    if nude_class_index is not None and  yolo_number_index is not None and nude_class_index <  yolo_number_index:
        # 提取 "nude_class" 和 "object" 之间的数据
        nude_data = df.iloc[nude_class_index + 1: yolo_number_index]
        # 提取名称和位置数据
        nude_class = nude_data.iloc[:, 0].tolist()[:5]
        nude_position = nude_data.iloc[:, 2].tolist()[:5]
    nude_positions_processed = []
    if len(nude_class) > 0:
        nude_positions_processed = [process_nude_position(pos) for pos in nude_position]

    # yolo目标检测部分
    if yolo_index is not None and license_index is not None and yolo_index < license_index:
        yolo_data = df.iloc[yolo_index + 1: license_index]
        # yolo名称和位置数据
        yolo_name = yolo_data.iloc[:, 0].tolist()
        # 提取每个节点对应的位置信息列表
        yolo_position = []
        for _, row in yolo_data.iterrows():
            yolo_position.append([row['onj'], row['sxmin'], row['symin'], row['sxmax']])
        # 计数YOLO节点
        for name, pos in zip(yolo_name, yolo_position):
            if name not in yolo_count_dict:
                yolo_count_dict[name] = {'count': 1, 'position': pos}
            else:
                yolo_count_dict[name]['count'] += 1


    # 特殊目标检测部分
    if license_index is not None and license_index < len(df) - 1:
        license_data = df.iloc[license_index + 1:]
        # 特殊检测的名称和位置数据
        license_name = license_data.iloc[:, 0].tolist()[:5]
        # 提取每个节点对应的位置信息列表
        license_position = []
        for _, row in license_data.iterrows():
            license_position.append([row['onj'], row['sxmin'], row['symin'], row['sxmax']])
        license_position = license_position[:5]


    # 提取sc部分节点名称列表
    for _, row in sc_data.iterrows():
        node_list.append(row['sub'])
        node_list.append(row['onj'])
        relationship_list.append(row['rel'])

    # # 提取sc部分关系名称列表
    # relationship_list = list(sc_data['rel'])
    # sc部分三元组数量
    sc_num = len(relationship_list)

    # 提取每个节点对应的位置信息列表
    node_positions = []
    node_positions.append(['0','0','0','0'])  #场景节点
    for _, row in sc_data.iterrows():
        # node_positions.append([row['sxmin'][7:15], row['symin'][7:15], row['sxmax'][7:15], row['symax'][7:15]])
        # node_positions.append([row['oxmin'][7:15], row['oymin'][7:15], row['oxmax'][7:15], row['oymax'][7:15]])
        node_positions.append([match_tensor(row['sxmin']),match_tensor(row['symin']),match_tensor(row['sxmax']),match_tensor(row['symax'])])
        node_positions.append([match_tensor(row['oxmin']),match_tensor(row['oymin']),match_tensor(row['oxmax']),match_tensor(row['oymax'])])

    # 加入yolo部分
    yolo_num = 0  # 标识yolo检测节点个数
    if sc_num < max_rows:
        rest_num = max_rows-sc_num
        node_list = node_list + yolo_name[0:rest_num]
        node_positions = node_positions + yolo_position[0:rest_num]
        yolo_num = min(rest_num, len(yolo_name))
        for i in range(yolo_num):
            relationship_list.append('in')
    # 加入yolo计数部分
    yolo_count_num = 0 # 标识yolo数量关系的个数
    for name, value in yolo_count_dict.items():
        if value["count"] > 1:
            node_num = value["count"]
            node_list.append(name)
            node_list.append(f"{node_num}")
            node_positions.append(value["position"])
            node_positions.append(['0', '0', '0', '0'])
            relationship_list.append('number')
            yolo_count_num += 1

    # 映射后新的节点名称列表和位置信息列表
    node_list2 = []
    node_positions2 = []
    old_to_new_index_map= []

    for i, (node, pos) in enumerate(zip(node_list, node_positions)):
        found = False
        for j, (new_node, new_pos) in enumerate(zip(node_list2, node_positions2)):
            if pos == ['0', '0', '0', '0']:
                break
            # 如果是同一个节点，则不加入
            if node == new_node or node == 'person':
                if is_same_node(pos, new_pos, tolerance):
                    old_to_new_index_map.append((i, j))
                    found = True
                    break
        if not found:
            node_list2.append(node)
            node_positions2.append(pos)
            old_to_new_index_map.append((i, len(node_list2) - 1))

    relationship_pairs = []
    # 根据新旧节点顺序的映射调整关系对应节点的顺序
    for i in range(sc_num):
        node1_index_old = 2 * i + 1
        node2_index_old = 2 * (i+1)
        node1_index_new = next(new for old, new in old_to_new_index_map if old == node1_index_old)
        node2_index_new = next(new for old, new in old_to_new_index_map if old == node2_index_old)
        relationship_pairs.append((node1_index_new, node2_index_new, i))
    for i in range(yolo_num):
        node1_index_old = 2 * sc_num + i + 1
        node1_index_new = next(new for old, new in old_to_new_index_map if old == node1_index_old)
        relationship_pairs.append((node1_index_new, 0, sc_num + i))
    for i in range(yolo_count_num):
        node1_index_old = 2 * sc_num + yolo_num + 2*i + 1
        node2_index_old = 2 * sc_num + yolo_num + 2*(i+1)
        node1_index_new = next(new for old, new in old_to_new_index_map if old == node1_index_old)
        node2_index_new = next(new for old, new in old_to_new_index_map if old == node2_index_old)
        relationship_pairs.append((node1_index_new, node2_index_new, sc_num + yolo_num + i))

    # 加入nude和license节点及关系
    for name, pos in zip(nude_class, nude_positions_processed):
        nude_index = len(node_list2)
        node_list2.append(name)
        node_positions2.append(pos)
        # 将与 node_list2[0] 的关系加入 relationship_pairs 和 relationship_list
        relationship_pairs.append((nude_index, 0, len(relationship_list)))
        relationship_list.append('nude')
    for l_name, l_pos in zip(license_name, license_position):
        license_index = len(node_list2)
        node_list2.append(l_name)
        node_positions2.append(l_pos)
        relationship_pairs.append((license_index, 0, len(relationship_list)))
        relationship_list.append('in')

    return node_list2, relationship_list, node_positions2, relationship_pairs


# 定义绘制矩形框的函数
def draw_boxes(image_path, node_list, node_position, output_path):
    image = cv2.imread(image_path)
    for name, pos in zip(node_list, node_position):
        x1, y1, x2, y2 = map(float, pos)  # 先转换为浮点数
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 4)
        cv2.putText(image, name, (x2, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0 , 0), 3)
    cv2.imwrite(output_path, image)


def load_labels(labels_path):
    """ Load labels from a JSON file and return a dictionary mapping image filenames to labels. """
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels_data = json.load(f)

    label = 0 if len(labels_data["labels"]) == 1 and labels_data["labels"][0] == "a0_safe" else 1
    return label


def load_labels_v2(labels_path, filename):
    try:
        with open(labels_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row[0] == filename:  # 假设第一列是文件名
                    if row[1] == "Public" or row[1] == "public":
                        label = 0
                    else:
                        label = 1
                    return label     # 返回第二列标签
    except Exception as e:
        print(f"读取文件时出错: {e}")
    return None


def main(args):
    dataset_dir = os.path.join(f"{args.data_path}", f"{args.dataset}")
    folder_path = os.path.join(dataset_dir, f"SC/{args.data_type}")
    image_path = os.path.join(dataset_dir, f"images/{args.data_type}")
    json_output_path = os.path.join(dataset_dir, f"{args.data_type}_detect.json")
    label_path = os.path.join(dataset_dir, f"label/{args.data_type}")

    # 获取文件夹中的所有CSV文件
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    # csv_files = ["2017_10356759.csv"]
    data_list = []
    for filename in tqdm(csv_files, desc="Processing CSV files"):
        csv_file_path = os.path.join(folder_path, filename)
        n_list, r_list, n_position, r_pair = process_csv(csv_file_path, 15, 25.0)
        # 输出结果
        # print(f"Processing file: {csv_file_path}")
        # print("关系名称列表:", r_list)
        # print("节点名称列表:", n_list)
        # print("节点位置信息列表:", n_position)
        # print("关系对应的节点顺序:", r_pair)

        base_filename = os.path.splitext(filename)[0]

        if args.dataset == "VISPR":
            label_file_path = os.path.join(label_path,filename.replace(".csv",".json"))
            label = load_labels(label_file_path)
        if args.dataset == "privacy_alert_v2":
            label_file_path = os.path.join(dataset_dir,f"Dataset_split/{args.data_type}_with_labels_2classes.csv")
            label = load_labels_v2(label_file_path, base_filename)
        if args.dataset == "picalert":
            label_file_path = os.path.join(dataset_dir, "privacysetting.csv")
            label = load_labels_v2(label_file_path, base_filename)
        data_entry = {
            "filename": base_filename,
            "n_list": n_list,
            "r_list": r_list,
            "n_position": n_position,
            "r_pair": r_pair,
            "label": label
        }
        data_list.append(data_entry)

        # # 图像文件路径
        # image_file_path = os.path.join(image_path, filename.replace(".csv", ".jpg"))
        #
        # # 用输出的结果给图片绘制矩形框，来测试输出结果是否合理。
        # output_file_path = os.path.join(image_path, "annotate", filename.replace(".csv", "_annotated.jpg"))
        # # 确保输出目录存在
        # os.makedirs(os.path.join(image_path, "annotate"), exist_ok=True)
        # # 绘制矩形框
        # draw_boxes(image_file_path, n_list, n_position, output_file_path)
        # print(f"Annotated image saved to: {output_file_path}")

    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    print(f"Data has been saved to {json_output_path}")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)