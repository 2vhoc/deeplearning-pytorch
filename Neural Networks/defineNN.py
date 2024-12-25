import torch, torchvision
import torch.nn as nn


class buildNN(nn.Module):
    def __init__(self, numClasses):
        super().__init__()
        self.ft = nn.Flatten()
        self.models = nn.Sequential(
            nn.Linear(in_features=32*32*3, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=numClasses),
        )

    def forward(self, x):
        x = self.ft(x)
        x = self.models(x)
        print(x.shape)
        return x



numClasses = 10
x = torch.randn(8, 3, 32, 32)
Model = buildNN(numClasses=numClasses)
Model(x)