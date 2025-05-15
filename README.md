
  <h1 align="center">Hierarchical Feature Fusion and Dynamic Multi-Scale Attention for Precision PCB Defect Detection</h1>
  
  <p align="center">Lei Zhang, Xiaoqian Zhang, Shukai Yang, Quan Feng, Yufeng Chen, Jie Zhang</p>

## News
Our following work "Hierarchical Feature Fusion and Dynamic Multi-Scale Attention for Precision PCB Defect Detection" is being submitted to The Visual Computer.

## Abstract
Printed circuit board (PCB) defect detection faces challenges due to complex industrial environments and the difficulty in detecting minute defects. This study introduces the DCAM-RTDETR framework, which integrates hierarchical feature fusion and dynamic multi-scale attention to enhance defect detection accuracy. The framework employs a Cascaded Adaptive Multi-Scale Fusion (CAMF) module for progressive feature refinement and spatial saliency-guided dynamic weighting, focusing on critical defect regions. Additionally, a Dynamic Hybrid-Scale Attention Collaborative Network (DHACN) combines local multi-core attention with global multi-scale convolution to efficiently detect micro-defects. Experimental results on the DeepPCB and PCB-DATASET datasets demonstrate superior performance, with mAP50/mAP50-95 reaching 99%/81.6% and 95.6%/50.4%, respectively, outperforming existing methods. This research provides a novel technical solution for high-precision visual inspection, significantly enhancing defect detection accuracy in complex industrial scenarios.

## network
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/DCAM-RTDETR.png)
 DCAM-RTDETR model structure diagram, through the backbone network to extract S3, S4, S5 three layers of features, each layer of features through the CAMF processing, S5 through the Position Encoding as well as the AIFI intra-scale interaction to get the F5, F5, S4 and S3 into the subsequent FPN for feature fusion, and then after each layer of features through the DHACN processing, after entering to the final after PAN on feature map feature fusion to get feature maps P3, P4 and P5, the subsequent dynamic screening and optimization of the target query by IoU-aware Query Selection (IoU-aware Query Selection). Finally defect detection through Transformer Decoder as well as header. where RepBlock is a reparameterisable convolution.

![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/CAMF.png)
Structure of the CAMF. The module mainly consists of Cascaded Hierarchical Feature Generator (CHFG) and Saliency-Guided Dynamic Weighting Module (SGD-WM). MP stands for Maximum Pooling.  Adaptive Weight Layer is used to extract the weights of each layer.

![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/DHACN.png)
Structure of DHACN. The module mainly consists of local and global dual paths and a lightweight weight distribution network. In local attention, the output feature maps are weighted and fused with each local output feature map through dynamic weights. In global multi-scale attention, the outputs are mainly obtained by averaging, 1×1 convolution enables cross-channel feature integration through global correlation of channel dimensions.

## Experimental Results
### Quantitative result
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/table3.png)
Comparison with DeepPCB state-of-the-art methods.
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/table4.png)
Comparison with PCB-DATASET state-of-the-art methods.
### Visualization Results Comparison
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/deeppcbzong.png)
The visualization result on the left is from YOLOv10, while the one on the right is from our method.
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/pcb可视化.drawio.png)
Visualization of Our Method on the PCB-DATASET Dataset.
![image](https://github.com/democt-ct/DCAM-RTDETR/blob/main/yolov10.drawio.png)
Visualization of YOLOv10 on the PCB-DATASET Dataset.

## Getting Started
### Environment
1.git clone https://github.com/democt-ct/DCAM-RTDETR.git
2.Create a new conda environment and install dependencies:
CUDA=11.8
pip install -r requirements.txt

### train & infer
python tools/train.py

python tools/infer.py

### Datasets
DeepPCB:https://github.com/tangsanli5201/DeepPCB
PCB-DATASET:https://robotics.pkusz.edu.cn/resources/dataset/

## Citation
```
@article{zhang2025dcam,
  title={Hierarchical Feature Fusion and Dynamic Multi-Scale Attention for Precision PCB Defect Detection},
  author={Lei Zhang, Xiaoqian Zhang, Shukai Yang, Quan Feng, Yufeng Chen, Jie Zhang},
  journal={The Visual Computer},
  year={2025},
  publisher={Springer}
}
```
## Acknowledgment
A part of this code is adapted from the previous work: RT-DETR (https://github.com/lyuwenyu/RT-DETR).
