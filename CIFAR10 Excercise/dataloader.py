
import torch, torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import pickle
dataTraining = CIFAR10(root='./data', train=True, transform=ToTensor())
import cv2, os
import numpy as np
dataLoader = DataLoader(
    dataset=dataTraining,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    drop_last=True
)
images = None
for img, lb in dataLoader:
    images = img[0].numpy()
    images = np.transpose(images, (1, 2, 0))
cv2.imshow('images', images)
cv2.waitKey(0)
print("Completed")