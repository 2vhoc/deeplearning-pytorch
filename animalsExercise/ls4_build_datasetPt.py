import torch, torchvision, os, cv2
# if torch.cuda.is_available():
#     print("hello")
# else:
#     print("Not co cuda")
#

class buildDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.shapes = []
        for data in os.listdir(root):
            image = []
            label = []
            shape = []
            for imgs in os.listdir(os.path.join(root, data)):
                img = cv2.imread(os.path.join(root, data, imgs), 1)
                    # img = cv2.resize(img, (224, 224))
                image.append(img)
                label.append(str(data))
                shape.append(img.shape)
            self.images.extend(image)
            self.labels.extend(label)
            self.shapes.extend(shape)


    def __len__(self):
        return(len(self.labels))
    def __getitem__(self, index):
        image = self.images[index]
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        shape = self.shapes[index]
        return image, label, shape
# root = "./archive/raw-img"
# data = buildDataset(root)
# img, lb, shp = data.__getitem__(10023)
# print(img.shape, lb)
# cv2.imshow("img", img.reshape(shp))
# print(lb)
# cv2.waitKey(0)

