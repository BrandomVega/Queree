import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.transforms import ToPILImage

weights = MobileNet_V2_Weights.DEFAULT
model = mobilenet_v2(weights=weights)


import warnings
warnings.filterwarnings("ignore")

import torch
torch.manual_seed(42)


places_train_dataset = datasets.Places365(
    root="data",
    download=True,
    split="val",
    small=True,
    transform=ToTensor()
)

train_dataset = DataLoader(
    dataset=places_train_dataset,
    batch_size=1,
    shuffle=False
)

batch = next(iter(train_dataset))
print(batch[0].shape) 
X = batch[0]

model.eval()
with torch.no_grad():
    for i in range(10):
        y = model(X)
        pred = torch.argmax(y,1)
        print(weights.meta['categories'][pred.item()])

