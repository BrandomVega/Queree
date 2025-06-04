import torch
from torchvision import transforms
from gensim.models import KeyedVectors
from PIL import Image
import urllib.request
import time
import os
import json

relative_path = os.getcwd()

embeddig_path = relative_path + "/nlp/embeddings/"
model = 'glove-twitter-200'
vectors = KeyedVectors.load(embeddig_path + model + '.bin')

async def run_MoE_1_imagenet(websocket, imagepaths, tasks):
    bestScores = []
    total = len(imagepaths)

    print(f"Loading Model")
    model_path = relative_path+ '/MoE_1/MoE_1_model.pth'
    model = torch.load(model_path)
    
    print("Preprocessing images")
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
            transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Convert images to 3 color channels
    def convert_to_3_channels(image):
        if image.mode == "L":  # Grayscale
            return image.convert("RGB")
        elif image.mode == "RGBA":  # RGBA
            return image.convert("RGB")
        elif image.mode == "P":  # Palette with transparency
            return image.convert("RGB")
        elif image.mode == "RGB":  
            return image
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Process a batch of images
    def process_batch(imagepaths, start_idx, end_idx):
        input_tensors = []
        for filename,_ in imagepaths[start_idx:end_idx]:
            input_image = Image.open(filename)
            input_image = convert_to_3_channels(input_image)
            input_tensor = preprocess(input_image)
            input_tensors.append(input_tensor)

        input_batch = torch.stack(input_tensors).to(device)

        with torch.no_grad():
            output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities

    # Labels
    filename = relative_path + "/MoE_1/moe1_classes.txt"
    with open(filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Process images in batches
    batch_size = 20
    for start_idx in range(0, len(imagepaths), batch_size):
        # Update progress
        progress_value = 20 + (start_idx / total) * 10
        response = {
            "message": f"Running 1/4 vision expert: Objects - {start_idx}/{total}",
            "output": "nothing",
            "progress": f"{progress_value:.2f}"  
        }
        await websocket.send_text(json.dumps(response))

        end_idx = min(start_idx + batch_size, len(imagepaths))
        probabilities = process_batch(imagepaths, start_idx, end_idx)

        for i, prob in enumerate(probabilities):
            top5_prob, top5_catid = torch.topk(prob, 5)
            print(f"Image {start_idx + i + 1}: {imagepaths[start_idx+i][0]} {imagepaths[start_idx+i][1]}")

            class_counts = []
            for j in range(top5_prob.size(0)):
                full_class = categories[top5_catid[j]]
                first_word = full_class.strip().split()[0].split("-")[0].lower()
                prob_value = top5_prob[j].item()
                print(f"{first_word}: {prob_value:.4f}")
                class_counts.append(first_word)  

            print("\n")
            # Class scoring
            score = 0
            for task_class, target_count in tasks.items():
                best_sim = 0
                for image_class in class_counts:
                    if task_class in vectors.key_to_index and image_class in vectors.key_to_index:
                        similarity = vectors.similarity(task_class, image_class)
                        best_sim = max(best_sim, similarity)
                        print(f"{task_class} ~ {image_class} = {similarity}")
                score += best_sim + (1 if best_sim==1 else 0) + imagepaths[start_idx+i][1]
                print(f"    > Image score: {score} updated by '{task_class}' (best sim: {best_sim:.4f})")
           
            bestScores.append((imagepaths[start_idx + i][0], score))
            print(f"Total score for image: {score:.4f}\n")

    # Sort by best scorest
    bestScores.sort(key=lambda x: x[1], reverse=True)
    best_image_index = bestScores[0][0] if bestScores else None

    print(f"\nBest scores updated (all of them): ")
    print(bestScores)
    return bestScores