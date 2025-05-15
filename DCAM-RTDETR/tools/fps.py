import os
import sys
sys.path.append('/media/democt/82F4E8B9F4E8B119/zhuomian/detr code/rtdetr code/RT-DETR-main/rtdetr_pytorch/')
import time
import torch
from torch.cuda.amp import autocast

# 使用相对导入
from src.nn.backbone import PResNet
from src.zoo.rtdetr import HybridEncoder, RTDETRTransformer, RTDETR, RTDETRPostProcessor
from PIL import Image
from torchvision import transforms


def benchmark_rtdetr():
    device = "cuda"
    weights_path = "/media/democt/82F4E8B9F4E8B119/zhuomian/detr code/rtdetr code/output/1/JUBU-3,7/checkpoint0095.pth"
    data_dir = "/media/democt/82F4E8B9F4E8B119/zhuomian/detr code/rtdetr code/RT-DETR-main/PCB_DATASET/val"
    input_size = 640

    # 初始化模型
    backbone = PResNet(depth=18, freeze_norm=False, pretrained=False, return_idx=[1, 2, 3])
    encoder = HybridEncoder(in_channels=[128, 256, 512], enc_act='gelu', expansion=0.5,
                            eval_spatial_size=[input_size] * 2)
    decoder = RTDETRTransformer(num_classes=6, eval_idx=-1, num_decoder_layers=3, num_denoising=100,
                                feat_channels=[256] * 3, feat_strides=[8, 16, 32],
                                hidden_dim=256, num_levels=3, num_queries=300, eval_spatial_size=[input_size] * 2)
    model = RTDETR(backbone=backbone, encoder=encoder, decoder=decoder, multi_scale=[input_size]).to(device)
    model.load_state_dict(torch.load(weights_path)['model'])
    model.eval()
    postprocessor = RTDETRPostProcessor(num_classes=6, remap_mscoco_category=True)

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), antialias=True),
        transforms.ToTensor()
    ])

    # 加载数据（预加载到GPU）
    dataset = []
    for file in os.listdir(data_dir):
        if file.lower().endswith(('png', 'jpg', 'jpeg')):
            img = Image.open(os.path.join(data_dir, file)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            ori_size = torch.tensor(img.size[::-1], device=device)  # (W, H)
            dataset.append((x, ori_size))
    num_images = len(dataset)

    # 测试配置
    torch.cuda.synchronize()
    runs = 11
    fps_results = []

    with torch.no_grad():
        # 预热
        for _ in range(2):
            for x, ori in dataset:
                with autocast():
                    y = model(x)
                    postprocessor(y, ori)

        # 正式测试
        for run in range(runs):
            torch.cuda.synchronize()
            start_time = time.perf_counter()

            for x, ori in dataset:
                with autocast():
                    y = model(x)
                    postprocessor(y, ori)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time
            fps = num_images / elapsed
            fps_results.append(fps)
            print(f"Run {run + 1}/{runs} - FPS: {fps:.2f}")

    # 保存结果到权重所在目录
    save_dir = os.path.dirname(weights_path)
    save_path = os.path.join(save_dir, "rtdetr_fps_results.txt")

    with open(save_path, 'w') as f:
        f.write(f"RT-DETR Benchmark Results\n")
        f.write(f"Model: {os.path.basename(weights_path)}\n")
        f.write(f"Test data: {data_dir}\n")
        f.write(f"Input size: {input_size}x{input_size}\n")
        f.write(f"Device: {device}\n\n")

        f.write("Individual runs:\n")
        for idx, fps in enumerate(fps_results):
            f.write(f"Run {idx + 1}: {fps:.2f} FPS\n")

        valid_fps = fps_results[1:]
        avg_fps = sum(valid_fps) / len(valid_fps)
        f.write(f"\nAverage FPS (last 10 runs): {avg_fps:.2f}\n")
    print(f"平均FPS（后10次）: {avg_fps:.2f}")
    print(f"\n结果已保存至: {save_path}")


if __name__ == "__main__":
    benchmark_rtdetr()
