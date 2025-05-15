#
# import os
# import json
# import shutil
# import random
# import xml.etree.ElementTree as ET
# from sklearn.model_selection import train_test_split
#
# # 数据集路径
# dataset_path = r"D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET"
# annotations_path = os.path.join(dataset_path, "Annotations")
# images_path = os.path.join(dataset_path, "images")
#
# # 创建train, val, test文件夹,将总的数据集划分为三个集
# train_dir = os.path.join(dataset_path, "train")
# val_dir = os.path.join(dataset_path, "val")
#
# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)
#
#
# # 划分比例
# train_ratio = 0.8
# val_ratio = 0.2
#
# # 固定类别映射关系
# category_map = {
#     "missing_hole": 1,
#     "mouse_bite": 2,
#     "open_circuit": 3,
#     "short": 4,
#     "spur": 5,
#     "spurious_copper": 6
#
# }
#
#
# # 划分数据集
# image_files = os.listdir(images_path)
# train_files,val_files = train_test_split(image_files, test_size=val_ratio)
# # 移动文件到相应的目录
# for file in train_files:
#     shutil.move(os.path.join(images_path, file), os.path.join(train_dir, file))
# for file in val_files:
#     shutil.move(os.path.join(images_path, file), os.path.join(val_dir, file))
#
#
#
# def create_coco_structure():
#     return {
#         "info": {
#             "description": "Converted Dataset",
#             "version": "1.0",
#             "year": 2023,
#         },
#         "licenses": [],
#         "images": [],
#         "annotations": [],
#         "categories": []
#     }
#
#
# def parse_xml(annotation_path, image_id, annotation_id, category_map):
#     tree = ET.parse(annotation_path)
#     root = tree.getroot()
#
#     image_info = {
#         "file_name": root.find("filename").text,
#         "height": int(root.find("size/height").text),
#         "width": int(root.find("size/width").text),
#         "id": image_id,
#     }
#     annotations = []
#
#     for obj in root.findall("object"):
#         category_name = obj.find("name").text
#         category_id = category_map.get(category_name)
#
#
#         # 调试输出
#         print(f"Parsing object: {category_name}, ID: {category_id}")
#
#         if category_id is None:
#             continue
#
#         bndbox = obj.find("bndbox")
#         xmin = int(bndbox.find("xmin").text)
#         ymin = int(bndbox.find("ymin").text)
#         xmax = int(bndbox.find("xmax").text)
#         ymax = int(bndbox.find("ymax").text)
#
#         width = xmax - xmin
#         height = ymax - ymin
#
#         annotation = {
#             "id": annotation_id,
#             "image_id": image_id,
#             "category_id": category_id,
#             "bbox": [xmin, ymin, width, height],
#             "area": width * height,
#             "iscrowd": 0,
#         }
#         annotations.append(annotation)
#         annotation_id += 1
#
#     return image_info, annotations, annotation_id
#
#
# def voc_to_coco(xml_files, image_dir, output_json, category_map):
#     coco_data = create_coco_structure()
#     image_id = 1
#     annotation_id = 1
#
#     for xml_file in xml_files:
#         annotation_path = os.path.join(annotations_path, xml_file)
#         image_info, annotations, annotation_id = parse_xml(
#             annotation_path, image_id, annotation_id, category_map
#         )
#         coco_data["images"].append(image_info)
#         coco_data["annotations"].extend(annotations)
#         image_id += 1
#
#     # 按要求设置 `categories`
#     for category_name, category_id in category_map.items():
#         coco_data["categories"].append({
#             "id": category_id,
#             "name": category_name,
#             "supercategory": "none",
#         })
#
#     with open(output_json, "w") as f:
#         json.dump(coco_data, f, indent=4)
#
#
# # 分别生成train.json、val.json、test.json
# def generate_annotations(split_files, split_dir, output_json):
#     xml_files = [file.replace(".jpg", ".xml") for file in split_files]
#     voc_to_coco(xml_files, split_dir, output_json, category_map)
#
#
# generate_annotations(train_files, train_dir, os.path.join(dataset_path, "train.json"))
# generate_annotations(val_files, val_dir, os.path.join(dataset_path, "val.json"))
#
#
import os
import shutil

# 数据集路径
voc_annotations_path = "D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET\Annotations"  # VOC注释文件的路径
voc_train_images_path = r"D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET\train"  # 训练集图片路径
voc_val_images_path = r"D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET\val"  # 验证集图片路径
voc_train_annotations_path = r"D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET\train_label"  # 训练集XML保存路径
voc_val_annotations_path = r"D:\zhuomian\detr code\RT-DETR-main\PCB_DATASET\val_label"  # 验证集XML保存路径

# 创建输出文件夹
os.makedirs(voc_train_annotations_path, exist_ok=True)
os.makedirs(voc_val_annotations_path, exist_ok=True)

# 获取训练集和验证集的图像文件名
train_images = set(os.path.splitext(f)[0] for f in os.listdir(voc_train_images_path))
val_images = set(os.path.splitext(f)[0] for f in os.listdir(voc_val_images_path))

# 遍历annotations文件夹，将XML文件根据图像划分到训练集或验证集
for xml_file in os.listdir(voc_annotations_path):
    if xml_file.endswith('.xml'):
        # 获取图像文件名（去掉扩展名）
        image_name = os.path.splitext(xml_file)[0]

        # 判断该XML文件对应的是训练集还是验证集
        if image_name in train_images:
            shutil.copy(os.path.join(voc_annotations_path, xml_file),
                        os.path.join(voc_train_annotations_path, xml_file))
        elif image_name in val_images:
            shutil.copy(os.path.join(voc_annotations_path, xml_file), os.path.join(voc_val_annotations_path, xml_file))

print("XML 文件已经按图像分配到 train 和 val 文件夹中。")
