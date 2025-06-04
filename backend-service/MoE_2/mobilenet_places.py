import torch
import urllib.request
from torchvision import transforms
from gensim.models import KeyedVectors
from torch.autograd import Variable as V
from torch.nn import functional as F
import torchvision.models as models
from PIL import Image
import json
import time
import os
import re

relative_path = os.getcwd()
embeddig_path = relative_path + "/nlp/embeddings/"
model_path = relative_path + "/MoE_0/"+"yolov8n.pt"

model = 'glove-twitter-25'     
vectors = KeyedVectors.load(embeddig_path + model + '.bin')

async def run_MoE_2_places(websocket, imagepaths, tasks):
    bestScores = []
    total = len(imagepaths)
    
    print(f"Loading Model")
    model_file = f'/mobnet_places365.pth.tar'
    
    path = relative_path+'/MoE_2'+model_file
    model = models.__dict__['resnet18'](num_classes=365)
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    #model_path = relative_path+ '/MoE_2/model.pt'
    #model = torch.load(model_path, map_location=torch.device('cpu'))

    # Preprocessing steps
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert images to 3 channels
    def convert_to_3_channels(image):
        if image.mode == "L":  # Grayscale
            return image.convert("RGB")
        elif image.mode == "RGBA":  # RGBA
            return image.convert("RGB")
        elif image.mode == "P":  # Palette with transparency
            return image.convert("RGB")
        elif image.mode == "RGB":  # Already RGB
            return image
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    batch_size = 20

    # Process a batch of images
    def process_batch(imagepaths, start_idx, end_idx):
        input_tensors = []
        for filename,_ in imagepaths[start_idx:end_idx]:
            input_image = Image.open(filename)
            print(filename)
            input_image = convert_to_3_channels(input_image)
            input_tensor = preprocess(input_image)
            input_tensors.append(input_tensor)

        input_batch = torch.stack(input_tensors).to(device)

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities

    # Load the places classes
    filename = relative_path + "/MoE_2/categories_places365.txt"
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    print(f"Processing {len(imagepaths)}")

    # Process images in batches
    for start_idx in range(0, len(imagepaths), batch_size):
        progress_value = 40 + (start_idx / total) * 10
        response = {
            "message": f"Running 2/4 vision expert: Scenes - {start_idx}/{total}",
            "output": "nothing",
            "progress": f"{progress_value:.2f}"  
        }
        await websocket.send_text(json.dumps(response))

        end_idx = min(start_idx + batch_size, len(imagepaths))
        probabilities = process_batch(imagepaths, start_idx, end_idx)

        # top 5 categories per image in the batch
        for i, prob in enumerate(probabilities):
            top10_prob, top10_catid = torch.topk(prob, 5)
            print(f"Image {start_idx + i + 1}: {imagepaths[start_idx+i][0]}")
            
            class_counts = []
            for j in range(top10_prob.size(0)):
                full_class = categories[top10_catid[j]]
                prob_value = top10_prob[j].item()
                print(f"{full_class}: {prob_value:.4f}")
                class_counts.append(full_class)  # Collect top predicted class names

            # Scoring
            score = 0
            for task_class, target_count in tasks.items():
                best_sim = 0
                for image_class in class_counts:
                    if task_class in vectors.key_to_index and image_class in vectors.key_to_index:
                        similarity = vectors.similarity(task_class, image_class)
                        best_sim = max(best_sim, similarity)
                        print(f"    {task_class} ~ {image_class} = {similarity}")
                    else:
                        print(f"not found {task_class} o {image_class}")
                score += best_sim + (1 if best_sim==1 else 0) + imagepaths[start_idx+i][1] 
           
            bestScores.append((imagepaths[start_idx + i][0], score))
            print(f"Total score for image: {score:.4f}\n")

    # Sort 
    bestScores.sort(key=lambda x: x[1], reverse=True)
    best_image_index = bestScores[0][0] if bestScores else None

    bestScores = bestScores[:10]

    print(f"\nBest scores updated: ")
    print(bestScores)
    return bestScores