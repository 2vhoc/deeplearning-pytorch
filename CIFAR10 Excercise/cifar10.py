import torch, torchvision
from torchvision import transforms
import pickle, numpy as np, os
class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, train=True):
        if train:
            dataFiles = list(os.path.join(root, f"data_batch_{i}") for i in range(1, 6))
        else:
            dataFiles = list(os.path.join(root, "test_batch"))
        self.images = []
        self.labels = []
        for dataFile in dataFiles:
            with open(dataFile, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])
        print(len(self.images), len(self.labels))


    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        return image, label


datasets = Datasets(root="./cifar-10-batches-py", train=True)

image, label = datasets.__getitem__(234)
print(image.reshape(32,32, 3).shape)
print(label)