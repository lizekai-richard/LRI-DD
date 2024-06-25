import os
import torch
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchvision.transforms.functional import to_pil_image

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet18(pretrained=True).eval().to(device)

# 输入图像和保存CAM图像的根目录
input_root_dir = "/home/nus-wk/DATASET/ILSVRC/Data/CLS-LOC/train"
output_root_dir = "/home/nus-wk/activation_map_of_IN1K/maps_of_in1k"

# 需要展示的层
target_layers = ['layer1', 'layer2', 'layer3', 'layer4']

# 图像预处理
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# 创建数据集和数据加载器
dataset = datasets.ImageFolder(root=input_root_dir, transform=preprocess)
dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)  # 根据需要调整batch_size和num_workers


# 遍历数据加载器并生成CAM
total_batches = len(dataloader)
for batch_idx, batch in enumerate(tqdm(dataloader, total=total_batches, desc="Processing Batches")):
    inputs, targets = batch
    inputs = inputs.to(device)
    paths = [dataset.samples[i][0] for i in range(batch_idx * 256, batch_idx * 256 + len(inputs))]

    # import pdb; pdb.set_trace()
    # paths = [dataset.samples[i][0] for i in range(inputs.size(0))]
    # print(paths[0])
    # import pdb; pdb.set_trace()

    for target_layer in target_layers:
        with SmoothGradCAMpp(model, target_layer=target_layer) as cam_extractor:
            # 将预处理后的数据传入模型
            outputs = model(inputs)
            # 通过传入类别索引和模型输出获取CAM

            activation_maps = cam_extractor([outputs[i].argmax().item() for i in range(outputs.size(0))], outputs)
            for j in range(len(activation_maps[0])):

                # raw_cam_path = os.path.join(output_root_dir, os.path.relpath(paths[j], input_root_dir)[:-5], f"{target_layer}_raw_cam.png")


                # import pdb; pdb.set_trace()
                # plt.imshow(activation_maps[0][j].squeeze(0).cpu().numpy())
                cam_tensor = activation_maps[0][j].squeeze(0).cpu().numpy()
                cam_numpy_path = os.path.join(output_root_dir, os.path.relpath(paths[j], input_root_dir)[:-5], f"{target_layer}_raw_cam.npy")
                os.makedirs(os.path.dirname(cam_numpy_path), exist_ok=True)
                np.save(cam_numpy_path, cam_tensor)
                print(cam_numpy_path)
                #################


    # 打印当前显存占用
    print(f"Batch {batch_idx + 1}/{total_batches}, Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")
