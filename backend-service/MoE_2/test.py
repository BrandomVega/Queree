import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import os

arch = 'mobnet'
model_file = f'{arch}_places365.pth.tar'
model = models.__dict__['resnet18'](num_classes=365)
checkpoint = torch.load(model_file, map_location='cpu')
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# class labels
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)
print(classes)

img_name = '/home/tudv/Documents/QUEREE/backend-service/MoE_3/bus.jpg'

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

# output the prediction
print('{} prediction on {}'.format(arch,img_name))
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))