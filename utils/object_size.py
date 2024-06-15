import os
import transformers
import requests
import torch
import cv2
import numpy as np
import torchvision.transforms.functional as F
from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask


def get_imagenet_1k(data_path):
    channel = 3
    num_classes = 1000

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=mean, std=std)
    ])

    dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)  # no augmentation
    dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

    # dst_train = datasets.ImageFolder(os.path.join(data_path, "train"))
    # dst_test = datasets.ImageFolder(os.path.join(data_path, "val"))

    class_names = dst_train.classes
    class_map = {x: x for x in range(num_classes)}

    return dst_train, dst_test, num_classes


def get_model(model_path):

    processor = MobileViTImageProcessor.from_pretrained(model_path)
    model = MobileViTV2ForImageClassification.from_pretrained(model_path)
    return processor, model


def object_size_by_class(dst_train, class_num, model, batch_size=32, device='cuda:0'):
    train_indices = torch.arange(len(dst_train))
    targets = torch.tensor(dst_train.targets, dtype=torch.long)
    class_indices = train_indices[targets == class_num]

    class_subset = Subset(dst_train, class_indices)

    data_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)
    total_batches = len(data_loader)
    
    target_layers = [model.mobilevitv2.encoder.layer[3]]
    cam = SmoothGradCAMpp(model=model, target_layers=target_layers)

    total_object_size = 0
    for batch_idx, (images, labels) in enumerate(tqdm(data_loader, total=total_batches, desc="Processing Batches")):
        images = images.to(device)
        paths = [class_subset.samples[i][0] for i in range(batch_idx * batch_size, batch_idx * batch_size + len(images))]
        
        outputs = model(images)
        activation_maps = cam(labels.tolist(), outputs)

        for i in range(len(activation_maps[0])):
            image = Image.open(paths[i]).convert('RGB')
            cam_img = F.to_pil_image(activation_maps[0][i].squeeze(0), mode='F')
            cam_img = cam_img.resize(image.size)
            result = overlay_mask(image, cam_img, alpha=0.5)
            print(result.size())
    # return cam_images
            


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# model_path = "/home/kwang/big_space/lzk/mobilevit-v2-imagenet1k"
# processor, model = get_feature_extractor(model_path)
# print(model.mobilevitv2.encoder.layer[3])
if __name__ == '__main__':
    device = torch.device("cuda:0")
    data_path = "/home/kwang/big_space/datasets/imagenet"
    dst_train, dst_test, num_classes = get_imagenet_1k(data_path)

    model_path = "/home/kwang/big_space/lzk/mobilevit-v2-imagenet1k"
    processor, model = get_model(model_path)
    # processor = processor.to(device)
    model = model.to(device).eval()

    for class_label in range(num_classes):
        object_size_by_class(dst_train, class_label, model, device=device)

