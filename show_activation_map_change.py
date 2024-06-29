import os
import torch
import torchvision
import kornia as K
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image


def recover_image():
    # zca_trans = get_zca_trans()
    logged_files_dir = "./distill/logged_files/CIFAR10/10/ConvNet/cifar10_ipc10_loss_scaled_activation_alter_100_dyn_map/Normal"
    for it in range(0, 20001, 500):
        images_path = os.path.join(logged_files_dir, "images_zca_{}.pt".format(it))
        if not os.path.exists(images_path):
            continue
        syn_images = torch.load(images_path)
        std = torch.std(syn_images)
        mean = torch.mean(syn_images)
        syn_images = torch.clip(syn_images, min = mean - 2.5 * std, max = mean + 2.5 * std)
        syn_images = torch.repeat_interleave(syn_images, repeats=4, dim=2)
        syn_images = torch.repeat_interleave(syn_images, repeats=4, dim=3)
        plt.imshow(to_pil_image(syn_images[0].detach()))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("test.png")
        plt.close()
        # grid = torchvision.utils.make_grid(syn_images, nrow=10, normalize=True, scale_each=True)

        # if not isinstance(grid, list):
        #     imgs = [grid]
        # fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
        # for i, img in enumerate(imgs):
        #     img = img.detach()
        #     img = to_pil_image(img)
        #     axs[0, i].imshow(np.asarray(img))
        #     axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        # plt.savefig("test.png")
        # break


def get_zca_trans():
    data_path = "/home/kwang/big_space/lzk/dataset/"
    transform = transforms.Compose([transforms.ToTensor()])

    dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform)  # no augmentation
    images = []
    labels = []
    for i in tqdm(range(len(dst_train))):
        im, lab = dst_train[i]
        images.append(im)
        labels.append(lab)
    images = torch.stack(images, dim=0).to("cpu")
    labels = torch.tensor(labels, dtype=torch.long, device="cpu")
    zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
    zca.fit(images)

    return zca

def main():

    logged_files_dir = "./distill/logged_files/CIFAR10/50/ConvNet/cifar10_ipc50_loss_scaled_activation_alter_100_dyn_conv_map/Normal"
    save_dir = "./overlay_results"
    num_classes = 10
    ipc = 50
    model = "convnetd3"

    # zca_trans = get_zca_trans()

    save_dir_shallow = os.path.join(save_dir, "CIFAR10", "ipc{}".format(ipc), model)
    save_dir_deep = os.path.join(save_dir, "CIFAR10", "ipc{}".format(ipc), model)

    if not os.path.exists(save_dir_shallow):
        os.makedirs(save_dir_shallow)
    if not os.path.exists(save_dir_deep):
        os.makedirs(save_dir_deep)

    for it in tqdm(range(0, 20001, 500)):
        images_path = os.path.join(logged_files_dir, "images_zca_{}.pt".format(it))
        if not os.path.exists(images_path):
            print("Not exists...")
            continue
        activation_maps_shallow_path = os.path.join(logged_files_dir, "activation_maps_shallow_{}.pt".format(it))
        activation_maps_deep_path = os.path.join(logged_files_dir, "activation_maps_deep_{}.pt".format(it))
        
        syn_images = torch.load(images_path)
        activation_maps_shallow = torch.load(activation_maps_shallow_path)
        activation_maps_deep = torch.load(activation_maps_deep_path)

        std = torch.std(syn_images)
        mean = torch.mean(syn_images)
        syn_images = torch.clip(syn_images, min = mean - 2.5 * std, max = mean + 2.5 * std)

        overlay_images_shallow = []
        overlay_images_deep = []
        for i, img in enumerate(syn_images):
            img = to_pil_image(img)
            activation_map_shallow = to_pil_image(activation_maps_shallow[i], mode='F')
            activation_map_deep = to_pil_image(activation_maps_deep[i], mode='F')
            cam_img_shallow = overlay_mask(img, activation_map_shallow, alpha=0.5)
            cam_img_deep = overlay_mask(img, activation_map_deep, alpha=0.5)
            overlay_images_shallow.append(transforms.ToTensor()(cam_img_shallow))
            overlay_images_deep.append(transforms.ToTensor()(cam_img_deep))
        
        overlay_images_shallow = torch.stack(overlay_images_shallow)
        overlay_images_deep = torch.stack(overlay_images_deep)

        overlay_images_shallow = torch.repeat_interleave(overlay_images_shallow, repeats=4, dim=2)
        overlay_images_shallow = torch.repeat_interleave(overlay_images_shallow, repeats=4, dim=3)
        overlay_images_deep = torch.repeat_interleave(overlay_images_deep, repeats=4, dim=2)
        overlay_images_deep = torch.repeat_interleave(overlay_images_deep, repeats=4, dim=3)

        grid_shallow = torchvision.utils.make_grid(overlay_images_shallow, nrow=10, normalize=True, scale_each=True)
        grid_deep = torchvision.utils.make_grid(overlay_images_deep, nrow=10, normalize=True, scale_each=True)

        
        if not isinstance(grid_shallow, list):
            imgs_shallow = [grid_shallow]
        
        fig, axs = plt.subplots(ncols=len(imgs_shallow), squeeze=False)
        for i, img in enumerate(imgs_shallow):
            img = img.detach()
            img = to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(os.path.join(save_dir_shallow, "overlay_shallow_it{}.pdf".format(it)))
        plt.close()

        if not isinstance(grid_deep, list):
            imgs_deep = [grid_deep]
        
        fig, axs = plt.subplots(ncols=len(imgs_deep), squeeze=False)
        for i, img in enumerate(imgs_deep):
            img = img.detach()
            img = to_pil_image(img)
            axs[0, i].imshow(np.asarray(img))
            axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        plt.savefig(os.path.join(save_dir_shallow, "overlay_deep_it{}.pdf".format(it)))
        plt.close()
        

if __name__ == '__main__':
    # recover_image()
    main()