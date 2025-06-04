import os
import gc
import time
from tqdm import tqdm
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import pandas 
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

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

cudnn.benchmark = True
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

import mlflow
import mlflow.pytorch
from mlflow.models import infer_signature

#==========DATASET=====================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4577, 0.4412, 0.4080], [0.2329, 0.2303, 0.2400])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.4577, 0.4412, 0.4080], [0.2329, 0.2303, 0.2400])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

places_train_dataset = Places365(
    root="places365_standard",
    download=False,
    split="train-standard",
    small=True,
    transform=data_transforms['train']
)

BATCH_SIZE = 64
TRAIN_SIZE = 0.08
VAL_SIZE = 0.01

total_size = len(places_train_dataset)
train_size = int(TRAIN_SIZE * total_size)
val_size = int(VAL_SIZE * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(places_train_dataset, [train_size, val_size, test_size])

train_dataset.dataset.transform = data_transforms['train']
val_dataset.dataset.transform = data_transforms['val']
test_dataset.dataset.transform = data_transforms['test']

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

dataloaders = {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader
}
dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset),
    'test': len(test_dataset)
}
class_names = places_train_dataset.classes

print(f"==="*10)
x,y = next(iter(dataloaders['val']))
print(f"Dataset samples:\n    {dataset_sizes}")
print(f"Batch size: {BATCH_SIZE}")
iterations_batch = dataset_sizes['train']/BATCH_SIZE
print(f"Iteraciones en train: {round(iterations_batch,1)}")
print(f"Device: {device}")
print(f"==="*10)

print(f"Starting...")

bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'

def top5acc(outputs, targets, k=5):
    topk_preds = torch.topk(outputs, k, dim=1).indices
    correct = topk_preds.eq(targets.view(-1,1))
    topkacc = correct.any(dim=1).float().mean().item()
    return topkacc

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    for epoch in tqdm(range(num_epochs), desc="Training", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]', leave=True, colour='cyan'):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            top5_acc_total = 0.0

            all_preds = []
            all_labels = []

            for inputs, labels in tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Phase", leave=False, colour="black"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    preds = torch.argmax(outputs, dim=1)
                
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                top5_acc = top5acc(outputs, labels, k=5)
                top5_acc_total += top5_acc * inputs.size(0)
            
                
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            accuracy_ = accuracy_score(all_labels, all_preds)
            precision_ = precision_score(all_labels, all_preds, average='macro', zero_division=0.0)
            recall_ = recall_score(all_labels, all_preds, average='macro')
            f1_score_ = f1_score(all_labels, all_preds, average='macro')

            top5_accuracy = top5_acc_total/dataset_sizes[phase]

            epoch_loss = running_loss / dataset_sizes[phase]
            #epoch_acc = running_corrects.double() / dataset_sizes[phase]

            mlflow.log_metric(f"Loss: {phase}", epoch_loss, epoch)
            #mlflow.log_metric(f"Accuracy: {phase}", epoch_acc, epoch)
            mlflow.log_metric(f"Accuracy: {phase}", accuracy_, epoch)
            mlflow.log_metric(f"Precision: {phase}", precision_, epoch)
            mlflow.log_metric(f"Recall: {phase}", recall_, epoch)
            mlflow.log_metric(f"F1_Score: {phase}", f1_score_, epoch)
            mlflow.log_metric(f"Top5_Acc: {phase}", top5_accuracy, epoch)

            #mlflow.log_metric(f"running_correct_{phase}", running_corrects, epoch)
            #mlflow.log_metric(f"running_correct_{phase}", running_corrects, epoch)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'])




    torch.save(model, "places_model.pt") 

def test_model(model_path, criterion, optimizer, scheduler, num_epochs=25):
    phase = "val"
    print(f"Testing Model with test dataset")
    model = torch.load(model_path, weights_only=False)
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []
    top5_acc_total = 0.0

    for inputs, labels in dataloaders[phase]:  
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        preds = torch.argmax(outputs, dim=1)
                
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        top5_acc = top5acc(outputs, labels, k=5)
        top5_acc_total += top5_acc * inputs.size(0)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy_ = accuracy_score(all_labels, all_preds)
    precision_ = precision_score(all_labels, all_preds, average='macro', zero_division=0.0)
    recall_ = recall_score(all_labels, all_preds, average='macro')
    f1_score_ = f1_score(all_labels, all_preds, average='macro')
    top5_accuracy = top5_acc_total/dataset_sizes[phase]

    print(f"Results of test:")
    print(f"    > Accuracy: {accuracy_}")
    print(f"    > Precision: {precision_}")
    print(f"    > Recall: {recall_}")
    print(f"    > F1_Score: {f1_score_}")
    print(f"    > Top5-Accuracy: {top5_accuracy}")
    


# =======Model=======
import torchvision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from torchvision.transforms import ToPILImage
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True

for param in model.features[-10:].parameters():
    param.requires_grad = True

num_classes = len(class_names)
lastconv_input_channels = model.features[-1][0].in_channels
lastconv_output_channels = 6*lastconv_input_channels

model.classifier = nn.Sequential(
    nn.Linear(lastconv_output_channels, 1280),
    nn.Hardswish(inplace=True),
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, num_classes),
)

#weights = MobileNet_V2_Weights.DEFAULT
#model = mobilenet_v2(weights=weights)

#for param in model.parameters():
#    param.requires_grad = False
    
#for param in model.features[18].parameters():
#    param.requires_grad = True


#for param in model.features[17].parameters():
#    param.requires_grad = True

#for param in model.features[16].parameters():
#    param.requires_grad = True


#num_ftrs = model.classifier[1].in_features
#model.classifier[1] = nn.Linear(num_ftrs, len(class_names))

model = model.to(device=device)

# mlflow setup
PORT = 4000
#os.system("mlflow server --host localhost --port 4000")
print(f"Connecting to tracking server at {PORT}",end=' ')
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri(uri=f"http://localhost:{PORT}")
mlflow.set_experiment("Training")
print(f"Done")
print("==="*10)

# hyperparameters
lr = 0.06
momentum = 0.9
weight_decay = 1e-5
num_epochs = 40
model_path = "places_model.pt"
# Track experiments. Log model not implemented. 
with mlflow.start_run():
    mlflow.log_param("criterion", "CrossEntropyLoss")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("lr", lr)
    mlflow.log_param("momentum", momentum)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_text("train_size",str(dataset_sizes['train']))
    mlflow.log_text("val_size",str(dataset_sizes['val']))
    mlflow.log_text("test_size",str(dataset_sizes['test']))
    mlflow.log_param("Batch_size", BATCH_SIZE)
    mlflow.log_param("train_percentage", TRAIN_SIZE)
    mlflow.log_param("val_percentage", VAL_SIZE)
    mlflow.log_param("model_path", model_path)

    criterion = nn.CrossEntropyLoss()
    #optimizer  = torch.optim.RMSprop(model.parameters(), lr=0.1, momentum=0.9, weight_decay=weight_decay)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    train_model(model, criterion, optimizer, None, num_epochs=num_epochs)
    test_model(model_path, criterion, None, None, None)

gc.collect() 
torch.cuda.empty_cache()
