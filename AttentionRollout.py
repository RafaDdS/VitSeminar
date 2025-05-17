import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, ToTensor

from timm.models import create_model

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests

from tqdm.auto import tqdm
import os
import warnings

import timm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def to_tensor(img):
    transform_fn = Compose([Resize(249, 3),
                            CenterCrop(224),
                            ToTensor(),
                            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_fn(img)


def show_img(img, cmap=None):
    img = np.asarray(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()


def show_img2(img1, img2, alpha=0.8, cmap=None):
    img1 = np.asarray(img1)
    img2 = np.asarray(img2)

    plt.figure(figsize=(10, 10))
    plt.imshow(img1, cmap=cmap)
    plt.imshow(img2, alpha=alpha, cmap=cmap)
    plt.axis('off')
    plt.show()


def my_forward_wrapper(attn_obj):
    def my_forward(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(
            B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 2:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward


def plot_set(image, attn_map, cls_weight, img_resized, cls_resized):
    # Create a figure and subplot grid
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

    # Plot the images on the subplots
    axs[0, 0].imshow(image)
    axs[0, 0].set_title('Image')

    axs[0, 1].imshow(attn_map)
    axs[0, 1].set_title('Attention Map')

    axs[0, 2].imshow(cls_weight)
    axs[0, 2].set_title('Class Weight')

    axs[1, 0].imshow(img_resized)
    axs[1, 0].set_title('Resized Image')

    # Overlay img_resized and cls_resized with alpha blending
    axs[1, 1].imshow(img_resized)
    axs[1, 1].imshow(cls_resized, alpha=0.7)
    axs[1, 1].set_title('Overlaid Images')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the figure
    plt.show()


def attention_rollout_function(attn_maps):
    attn_rollout = []
    I = torch.eye(attn_maps[0].shape[-1])  # Identity matrix
    prod = I
    for i, attn_map in enumerate(attn_maps):
        # Product of attention maps with identity matrix
        prod = prod @ (attn_map + I)

        prod = prod / prod.sum(dim=-1, keepdim=True)  # Normalize
        attn_rollout.append(prod)
    return attn_rollout


def plot_attention_maps(attn_maps, num_cols=4, main_title='Attention Maps'):
    num_attn_maps = len(attn_maps)

    # Calculate the number of rows and columns for the subplots
    num_rows = (num_attn_maps + num_cols - 1) // num_cols

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))

    # Set the main title
    fig.suptitle(main_title, fontsize=20)

    # Iterate over the attention maps and plot them in the subplots
    for i, attn_map in enumerate(attn_maps):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        ax.imshow(attn_map, cmap='viridis')
        ax.set_title(f'Attention Map {i+1}')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(num_attn_maps, num_rows * num_cols):
        row = j // num_cols
        col = j % num_cols
        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]
        ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Show the figure
    plt.show()


def plot_cls_weights(cls_weights, img_resized):
    num_cls_weights = len(cls_weights)

    # Calculate the number of rows and columns for the subplots
    num_cols = 4  # Adjust as needed
    num_rows = (num_cls_weights + num_cols - 1) // num_cols

    # Create a figure and a grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows))

    # Iterate over the cls_weights and plot them in the subplots
    for i, cls_weight in enumerate(cls_weights):
        row = i // num_cols
        col = i % num_cols

        if num_rows == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        cls_resized = F.interpolate(cls_weight.view(
            1, 1, 14, 14), (224, 224)).view(224, 224, 1)

        masked = img_resized*cls_resized
        masked -= masked.min()
        masked /= masked.max()
        
        ax.imshow(masked)
        ax.set_title(f'Class Weight {i+1}')
        ax.axis('off')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    # Show the figure
    plt.show()


def plot_heatmap(cls_weights, figsize=(12, 12)):
    # Flatten each tensor into a 1D tensor of length 196
    flattened_cls_weights = [tensor.reshape(-1) for tensor in cls_weights]

    # Create a 2D NumPy array from the flattened tensors
    data = np.array(flattened_cls_weights)

    # Create a figure with the specified size
    fig, ax = plt.subplots(figsize=figsize)

    # Create a heatmap with each row representing a tensor
    ax = sns.heatmap(data, ax=ax)

    # Set the y-axis tick labels to represent the tensor indices
    ax.set_yticks(range(len(cls_weights)))
    ax.set_yticklabels(range(len(cls_weights)))

    # Adjust the plot layout
    plt.tight_layout()

    # Show the heatmap
    plt.show()


# processor, model = load_model()
image = Image.open("images/Celular.jpg")
x = to_tensor(image)

model = create_model('deit_small_distilled_patch16_224',
                     pretrained=True).to(device)

for block in tqdm(model.blocks):
    block.attn.forward = my_forward_wrapper(block.attn)

y = model(x.unsqueeze(0).to(device))

attn_maps = []
cls_weights = []
for block in tqdm(model.blocks):
    attn_maps.append(block.attn.attn_map.max(dim=1).values.squeeze(0).detach())
    cls_weights.append(block.attn.cls_attn_map.mean(
        dim=1).view(14, 14).detach())

# Combine class scores of all blocks
cls_weight_combined = torch.prod(torch.stack(cls_weights), dim=0)
attn_maps_prod = torch.prod(torch.stack(attn_maps), dim=0)

img_resized = x.permute(1, 2, 0) * 0.5 + 0.5

img_resized_cpu = img_resized.cpu()

cls_weights_cpu = []
for i in range(12):
    cls_weight = cls_weights[i]
    cls_weight_cpu = cls_weight.cpu()
    cls_weights_cpu.append(cls_weight_cpu)

plot_cls_weights(cls_weights_cpu, img_resized_cpu)

_, ind = torch.topk(y, 5)

print(ind)
