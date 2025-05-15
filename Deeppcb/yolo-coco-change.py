import os
import json
from PIL import Image

# 设置数据集路径
output_dir = r"D:\zhuomian\detr code\RT-DETR-main\Deeppcb\dataset_coco\annotations"  # 输出的COCO格式数据集路径
dataset_path = r"D:\zhuomian\detr code\RT-DETR-main\Deeppcb\dataset_coco"  # 数据集根目录路径

# 类别映射
categories = [
    {"id": 1, "name": "1_open"},
    {"id": 2, "name": "2_short"},
    {"id": 3, "name": "3_mousebite"},
    {"id": 4, "name": "4_spur"},
    {"id": 5, "name": "5_copper"},
    {"id": 6, "name": "6_pin-hole"},
    # 添加更多类别
]

# YOLO格式转COCO格式的函数
def convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    width = width * img_width
    height = height * img_height
    return [x_min, y_min, width, height]

# 初始化COCO数据结构
def init_coco_format():
    return {
        "images": [],
        "annotations": [],
        "categories": categories
    }

# 处理每个数据集分区
for split in ['train', 'test', 'val']:
    coco_format = init_coco_format()
    annotation_id = 1

    # 更新为正确的图片和标签路径
    images_path = os.path.join(dataset_path, split, "images")  # 这里是图片的路径
    labels_path = os.path.join(dataset_path, split, "labels")  # 这里是标签的路径

    if not os.path.exists(images_path):
        print(f"警告: {images_path} 路径不存在！")
        continue

    if not os.path.exists(labels_path):
        print(f"警告: {labels_path} 路径不存在！")
        continue

    # 处理图像文件
    for img_name in os.listdir(images_path):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # 确保文件是图片格式
            img_path = os.path.join(images_path, img_name)
            label_path = os.path.join(labels_path, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

            if not os.path.exists(label_path):  # 如果没有找到对应的标签文件，跳过
                print(f"警告: 标签文件 {label_path} 不存在，跳过图片 {img_name}")
                continue

            img = Image.open(img_path)
            img_width, img_height = img.size
            image_info = {
                "file_name": img_name,
                "id": len(coco_format["images"]) + 1,
                "width": img_width,
                "height": img_height
            }
            coco_format["images"].append(image_info)

            # 处理标签文件
            with open(label_path, "r") as file:
                for line in file:
                    category_id, x_center, y_center, width, height = map(float, line.split())
                    bbox = convert_yolo_to_coco(x_center, y_center, width, height, img_width, img_height)
                    annotation = {
                        "id": annotation_id,
                        "image_id": image_info["id"],
                        "category_id": int(category_id) + 1,  # 假设类别从0开始，+1使其从1开始
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0
                    }
                    coco_format["annotations"].append(annotation)
                    annotation_id += 1

    # 为每个分区保存JSON文件
    json_output_path = os.path.join(output_dir, f"{split}_coco_format.json")
    with open(json_output_path, "w") as json_file:
        json.dump(coco_format, json_file, indent=4)

print("转换完成！")
