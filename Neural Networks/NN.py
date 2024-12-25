from ast import main
import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose
import pickle, numpy as np, os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#-----------------------------------------------------------------
class Datasets(torch.utils.data.Dataset):
    def __init__(self, root, train=True, tf=None):
        if train:
            dataFiles = list(os.path.join(root, f"data_batch_{i}") for i in range(1, 6))
        else:
            dataFiles = [(os.path.join(root, "test_batch"))]
        self.images = []
        self.labels = []
        for dataFile in dataFiles:
            with open(dataFile, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                self.images.extend(data[b'data'])
                self.labels.extend(data[b'labels'])

        self.tf=tf
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image = self.images[index]
        image = image.reshape(3, 32, 32).transpose(2, 0, 1)

        if tf:
            image = self.tf(image)
        label = self.labels[index]
        return image, label

# Build Model from scratch hehehehehee=))))
class NN(nn.Module):
    def __init__(self, numClasses=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, numClasses),
            nn.ReLU()
        )
    def forward(self, x):
        return self.model(x)

#-----------------------------------------------------------------
tf = Compose([
    ToTensor()
])
dataTrain = Datasets(root="/content/data/cifar-10-batches-py", train=True, tf=tf)
dataTest = Datasets(root="/content/data/cifar-10-batches-py", train=False, tf=tf)
trainLoader = DataLoader(
    dataset=dataTrain,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
testLoader = DataLoader(
    dataset=dataTest,
    batch_size=64,
    shuffle=True,
    num_workers=2,
    drop_last=True
)
# -----------------------------------------------------------
numClasses = 10
Model = NN(numClasses)
epochs = 100
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(
    Model.parameters(),
    lr=0.001,
    momentum=0.9
)

for epoch in range(epochs):
    Model.train()
    for iter, (imgs, lbs) in enumerate(trainLoader):
        opt.zero_grad()
        preds = Model(imgs)
        lossVal = criterion(preds, lbs)
        opt.zero_grad()
        lossVal.backward()
        opt.step()
        if iter % 100 == 0:
          print(f"Epoch {epoch}/{epochs} | Iter {iter}/{len(trainLoader)} | Loss {lossVal}")
#tinh accuracy :))))), ko phai chat gpt dau nhe:))
Model.eval()
predictions = []
LabelsVal = []
for epoch in range(epochs):
      for iter, (imgs, lbs) in enumerate(testLoader):
        LabelsVal.extend(lbs)
        with torch.no_grad():
            preds = Model(imgs)
            _, id = torch.max(preds, dim=1)
            predictions.extend(id)
            lossVal = criterion(preds, lbs)
predictions = [x.item() for x in predictions]
LabelsVal = [x.item() for x in LabelsVal]
print(predictions)
print("----------------------------------")
print(LabelsVal)
# _---------------------------------------------------_
print(classification_report(LabelsVal, predictions))
print("Completed")
exit(0)
