import torch

model = torch.load("model.pt", weights_only=False)
model.eval()
model.to("cpu")
from PIL import Image
from torchvision import transforms
from torchvision.datasets import Places365

places_train_dataset = Places365(
    root="places365_standard",
    download=False,
    split="train-standard",
    small=True
)
class_names = places_train_dataset.classes

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

import re

target = r"/home/me/Documentos/Queree/target"
import os
for file in os.listdir(target):
    img_path = os.path.join(target, file)
    img = Image.open(img_path)
    img.show()
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.to("cpu")
    with torch.no_grad():
        outputs = model(img_tensor)
        top_scores, top_indices = torch.topk(outputs, k=5)

    print(top_indices)
    print(img_path)
    for idx in top_indices[0]:
        classs = class_names[idx.item()]
        match = re.search(r'[^/]+$', classs)
        if match:
            print(match.group())
    wait = input("Continuar? ")

#FUNCIONA DECENTEMENTE
        
