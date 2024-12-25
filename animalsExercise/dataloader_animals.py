import numpy as np
import torch, torchvision, cv2, os
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
# from ls4_build_datasetPt import buildDataset

class buildDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        for idx, data in enumerate(os.listdir(root)):
            image = []
            label = []
            for img in os.listdir(os.path.join(root, data)):
                img = os.path.join(root, data, img)
                # img = cv2.resize(img, (224, 224))
                image.append(img)
                label.append(idx)
            self.images.extend(image)
            self.labels.extend(label)



    def __len__(self):
        return(len(self.labels))

    def __getitem__(self, index):
        imagePath = self.images[index]
        image = cv2.imread(imagePath)
        image = image[:, :, :3]
        print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


transform = Compose([
    ToTensor(),
    Resize((200, 200)),
    ])
dataTraining = buildDataset(root="./archive/raw-img", transform=transform)

dataLoader = DataLoader(
    dataset=dataTraining,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    drop_last=True
)

for imgs, lbs in dataLoader:
    print(imgs)
    print(lbs)
print("Completed")