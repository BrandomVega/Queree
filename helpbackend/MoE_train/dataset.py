import os
import gc
import time
from tqdm import tqdm
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import Places365
from torchvision.transforms import ToTensor

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

places_train_dataset = Places365(
    root="data",
    download=False,
    split="val",
    small=True,
    transform=ToTensor()
)


class_names = places_train_dataset.classes
class_idx = len(class_names)


BATCH_SIZE = 16
TRAIN_SIZE = 0.9
VAL_SIZE = 0.1

total_size = len(places_train_dataset)
train_size = int(TRAIN_SIZE * total_size)
val_size = int(VAL_SIZE * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(places_train_dataset, [train_size, val_size, test_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']
test_dataset.dataset.transform = data_transforms['test']

train_loader = DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset,batch_size = BATCH_SIZE,   shuffle=True)
test_loader = DataLoader(dataset=test_dataset,batch_size = BATCH_SIZE,   shuffle=False)


amount_classes = {key:0 for key in range(class_idx)}

#print(amount_classes)

#for img,label in places_train_dataset:
#    amount_classes[label]+=1
#print("TOTAL CLASES")
#print(amount_classes)
#for img,label in train_loader:
#    amount_classes[label.item()]+=1
#print(amount_classes)
#amount_classes = {key:0 for key in range(class_idx)}

#print(f"validacion")
#for img,label in val_loader:
#    amount_classes[label.item()]+=1

#print(amount_classes)

#mean = 0
#std = 0
#nb_samples = 0
#for data, labels in train_loader:
#    batch_samples = data.size(0)  # Get the batch size from the input tensor
#    data = data.view(batch_samples, data.size(1), -1)
#    mean += data.mean(2).sum(0)
#    std += data.std(2).sum(0)
#    nb_samples += batch_samples
    
#mean /= nb_samples
#std /= nb_samples

#print(mean)
#print(std)

#tensor([-0.1018, -0.0671, -0.0019])
#tensor([0.9440, 0.9499, 0.9707])

#tensor([0.4577, 0.4412, 0.4080])
#tensor([0.2329, 0.2303, 0.2400])
