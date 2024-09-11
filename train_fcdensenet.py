import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import DataLoader, TensorDataset
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress 
from utils import *

import torchvision.transforms as T
from torch.utils.data import Dataset
from densenet import *


data_folder = "../Database_134_Angiogram_128/"

# train_img_paths = [glob.glob(os.path.join(folder, '*/images/*.png')) for folder in train_folders]
# train_msk_paths = [glob.glob(os.path.join(folder, '*/annotations/*.png')) for folder in train_folders]


# train_img_paths = glob.glob(os.path.join(prjt_folder, train_folder, '*/*_train/images/*.png'))
msk_paths = glob.glob(os.path.join(data_folder, '*_gt.pgm'))
img_paths = glob.glob(os.path.join(data_folder, '*.pgm'))
# print(msk_paths)
# print(img_paths)
# Exclude files ending with '_gt.pgm'
img_paths = [file for file in img_paths if not file.endswith('_gt.pgm')]

def sort_files(file_list):
  sorted_list = sorted(file_list, key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[0]))
  return sorted_list

msk_paths = sort_files(msk_paths)
img_paths = sort_files(img_paths)

filt_std = 2 
resize_sz = 128


from scipy.ndimage import gaussian_filter
img=[]
msk=[]
print (len(img_paths))
for i in range(len(img_paths)):
  img.append(cv2.imread(img_paths[i], cv2.IMREAD_UNCHANGED))
  # print (np.array(
  #   Image.open (img_paths[i])).shape
  # )
  msk.append(cv2.imread(msk_paths[i], cv2.IMREAD_UNCHANGED))

def high_pass_filt(img, krn_sz=255, std=filt_std):
  """
  Applies a high-pass filter by subtracting a Gaussian-blurred image from the original.

  Args:
  - img (numpy array): The input image.
  - krn_sz (int): The size of the Gaussian kernel.
  - std (float): The standard deviation of the Gaussian blur.

  Returns:
  - high_pass_img (numpy array): The image with high-pass filtering applied.
  """
  # Apply Gaussian blur to the image to get the low-frequency component
  # low_freq = cv2.GaussianBlur(img, (krn_sz, krn_sz), std)
  low_freq = gaussian_filter(img, sigma=std)

  # Subtract the low-frequency component from the original image to get the high-frequency component
  high_pass_img = img - low_freq
  # high_pass_img = low_freq

  return high_pass_img

def data_process(img,msk, filt_std = 2, resize_sz=resize_sz):
  img = np.stack(img,0)
  msk = np.stack(msk,0)
  img = img.astype(np.float32)
  img = np.array(
    [high_pass_filt(im, filt_std) for im in img]
  )
  img = torch.tensor(img,dtype=torch.float32)
  # print (img.shape)
  msk = torch.tensor(msk>0.5,dtype=torch.float32)
  # Resize the tensor to [batch, 128, 128]
  img = F.interpolate(img.unsqueeze(1), size=(resize_sz, resize_sz), mode='bilinear', align_corners=False)
  msk = F.interpolate(msk.unsqueeze(1), size=(resize_sz, resize_sz), mode='bilinear', align_corners=False)
  return img,msk

ratio=6
bias = 5

print("datashape")
filt_img,filt_msk = data_process(img,msk)
print(filt_img.shape)
print(filt_msk.shape)
print()

def transf_img(img, ratio=ratio, bias=bias):
  img = (img + bias) * ratio
  return torch.clamp(img // 16, min=-8, max=8)+8

clamp_img = transf_img(filt_img)

# clamp_img = filt_img
print("clamp size")
print(clamp_img.shape)

train_ratio=0.8
val_ratio=0.
test_ratio=0.2
train_size=int(train_ratio*len(clamp_img))
val_size=int(val_ratio*len(clamp_img))
test_size=int(test_ratio*len(clamp_img))
train_img=clamp_img[:train_size]
train_msk=filt_msk[:train_size]
val_img=clamp_img[train_size:train_size+val_size]
val_msk=filt_msk[train_size:train_size+val_size]
test_img=clamp_img[train_size+val_size:]
test_msk=filt_msk[train_size+val_size:]

class SegmentationDataset(Dataset):
    def __init__(self, imgs, masks, transform=None):
        self.imgs = imgs
        self.masks = masks
        self.transform = transform
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx] 
        mask = self.masks[idx]  

        combined = torch.cat([img, mask], dim=0)
        if self.transform:
            
            combined = self.transform(combined)  # 应用变换

        img, mask = combined[0].unsqueeze(0), combined[1].unsqueeze(0)

        return img, mask.long()

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        # 创建高斯噪声
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # 将噪声加到图像上
        return tensor + noise
# 数据增强变换
def get_augmentation():
    return T.Compose([
        AddGaussianNoise(mean=0.0, std=0.05)
        #T.RandomHorizontalFlip(p=0.5),
        #T.RandomVerticalFlip(p=0.5),
        #T.ColorJitter(brightness=0.2, contrast=0.2, ),  # 亮度、对比度、饱和度和色调调整
        #T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),  # 高斯模糊，内核大小为5x9，sigma范围为0.1到5

    ])


if __name__ == "__main__":

    dataset = SegmentationDataset(train_img, train_msk, transform=None) #get_augmentation()
    dataset_test = SegmentationDataset(test_img, test_msk)
    batch = 10
    dataloader = DataLoader(dataset, batch_size = batch, shuffle = True)
    dataloader_test = DataLoader(
      dataset_test, batch_size = batch, shuffle = False
    )

    device = torch.device('cuda')

    model = FCDenseNet(
        in_channels=1, 
        down_blocks=(6, 12, 24, 16),  # DenseNet121 down block structure
        up_blocks=(16, 24, 12, 6),    # Symmetric up block structure
        bottleneck_layers=32,         # Bottleneck size, chosen based on DenseNet121
        growth_rate=32,               # Growth rate is 32 for DenseNet121
        out_chans_first_conv=32,      # Initial conv output channels
        n_classes=2)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        # Wrap the model with DataParallel to run across multiple GPUs
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(
        model.parameters(),
        lr = 1e-3,
        weight_decay = 1e-5
    )
    model_dir = 'model_8_300/'
    restore_model = False
    # print (model)
    train, test = train_model(
      model, dataloader=dataloader, dataloader_test=dataloader_test, 
      criterion=criterion, optimizer = optimiser, num_epochs = 100, 
      model_dir = model_dir, restore_model=restore_model
    )
    