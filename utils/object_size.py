import os
import transformers
import requests
import torch
import cv2
import numpy as np
from transformers import MobileViTImageProcessor, MobileViTV2ForImageClassification
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image


def get_imagenet_1k(data_path):
    channel = 3
    im_size = (128, 128)
    num_classes = 1000

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std),
                                    transforms.Resize(im_size),
                                    transforms.CenterCrop(im_size)])

    # dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform)  # no augmentation
    # dst_test = datasets.ImageFolder(os.path.join(data_path, "val"), transform=transform)

    dst_train = datasets.ImageFolder(os.path.join(data_path, "train"))
    dst_test = datasets.ImageFolder(os.path.join(data_path, "val"))

    class_names = dst_train.classes
    class_map = {x: x for x in range(num_classes)}

    return dst_train, dst_test, num_classes


def get_model(model_path):

    processor = MobileViTImageProcessor.from_pretrained(model_path)
    model = MobileViTV2ForImageClassification.from_pretrained(model_path)
    return processor, model


def object_size_by_class(dst_train, class_num, processor, model, batch_size=32, device='cuda:0'):
    train_indices = torch.arange(len(dst_train))
    targets = torch.tensor(dst_train.targets, dtype=torch.long)
    class_indices = train_indices[targets == class_num]

    class_subset = Subset(dst_train, class_indices)

    data_loader = DataLoader(class_subset, batch_size=batch_size, shuffle=False)

    with torch.inference_mode():
        target_layers = [model.mobilevitv2.encoder.layer[3]]
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
        # cam.batch_size = batch_size
        for image, label in tqdm(class_subset):
            image = np.array(image)
            image = cv2.resize(image, (224, 224))
            image = np.float32(image) / 255
            input_tensor = preprocess_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            print(input_tensor)
            targets = [ClassifierOutputTarget(label)]
            grayscale_cam = cam(input_tensor=input_tensor)
            cam_image = show_cam_on_image(image, grayscale_cam[0, :], use_rgb=True)
            print(cam_image)
            break
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
    model = model.to(device)

    object_size_by_class(dst_train, 0, processor, model)

