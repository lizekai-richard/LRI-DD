import os
import torch
import copy
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet18, get_model, ConvNeXt_Base_Weights
from torch.utils.data import DataLoader, TensorDataset
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from utils.utils_gsam import get_network


def demo():
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True).eval().to(device)

    # 输入图像和保存CAM图像的根目录
    input_root_dir = "/home/kwang/big_space/datasets/imagenet/train"
    output_root_dir = "/home/kwang/big_space/lzk/activation_map_of_IN1K/batch_in1k"

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
        paths = [dataset.samples[i][0] for i in range(batch_idx * 10, batch_idx * 10 + len(inputs))]
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

                    raw_cam_path = os.path.join(output_root_dir, os.path.relpath(paths[j], input_root_dir)[:-5], f"{target_layer}_raw_cam.png")
                    os.makedirs(os.path.dirname(raw_cam_path), exist_ok=True)

                    plt.imshow(activation_maps[0][j].squeeze(0).cpu().numpy())
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(raw_cam_path)
                    plt.close()

                    img = Image.open(paths[j]).convert('RGB')
                    cam_img = to_pil_image(activation_maps[0][j].squeeze(0), mode='F')
                    cam_img = cam_img.resize(img.size)
                    result = overlay_mask(img, cam_img, alpha=0.5)
                    overlay_cam_path = os.path.join(output_root_dir, os.path.relpath(paths[j], input_root_dir)[:-5], f"{target_layer}_overlay_cam.png")
                    plt.imshow(result)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(overlay_cam_path)
                    plt.close()
                    #################


        # 打印当前显存占用
        print(f"Batch {batch_idx + 1}/{total_batches}, Memory Allocated: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")


def get_activation_maps(image_syn, label_syn, num_classes, ipc, im_size, model_path=None, dataset='imagenet-1k'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dataset == 'ImageNet-1k':
        model = get_model("convnext_base", weights="IMAGENET1K_V1").eval().to(device)
        target_layers = [model.features[-1]]
    elif dataset == 'CIFAR10':
        model = get_network("ResNet18", 3, num_classes, im_size, dist=False).eval()
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        target_layers = ['layer4']
    elif dataset == 'CIFAR100':
        model = get_network("ResNet50", 3, num_classes, im_size, dist=False).eval()
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        target_layers = ['layer4']
    # transform = ConvNeXt_Base_Weights.IMAGENET1K_V1.transforms()
    
    cam_extractor = SmoothGradCAMpp(model=model, target_layer=target_layers)

    activation_maps_all = []

    for c in range(num_classes):
        image_syn_c = image_syn[c * ipc: (c + 1) * ipc]
        label_syn_c = label_syn[c * ipc: (c + 1) * ipc]
        # pil_images_c = [transforms.ToPILImage()(image_tensor).convert('RGB') for image_tensor in image_syn_c]
        
        dataset = TensorDataset(copy.deepcopy(image_syn_c.detach()), copy.deepcopy(label_syn_c.detach()))
        if ipc <= 50:
            batch_size = ipc
            
        else:
            batch_size = 32
            
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for images, labels in data_loader:
            images = images.to(device)

            outputs = model(images)
            activation_maps = cam_extractor(labels.argmax(dim=-1).tolist(), outputs)[0]
            resize = transforms.Resize(size=im_size, interpolation=transforms.InterpolationMode.BILINEAR)
            resized_activation_maps = resize(activation_maps)
            activation_maps_all.extend(resized_activation_maps)

            # for j in range(len(activation_maps[0])):
            #     # raw_cam_path = os.path.join(output_root_dir, os.path.relpath(paths[j], input_root_dir)[:-5], f"{target_layer}_raw_cam.png")
            #     # os.makedirs(os.path.dirname(raw_cam_path), exist_ok=True)
            #     img = pil_images_c[j]
            #     cam_img = to_pil_image(activation_maps[0][j].squeeze(0), mode='F')
            #     cam_img = cam_img.resize(img.size)
            #     result = overlay_mask(img, cam_img, alpha=0.5)
            #     activation_maps_all.append(transforms.ToTensor()(result))
    activation_maps_all = torch.stack(activation_maps_all, dim=0)
    return activation_maps_all