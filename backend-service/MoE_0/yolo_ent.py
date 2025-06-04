from gensim.models import KeyedVectors
from collections import Counter
from ultralytics import YOLO
import json
import os

# Load embeddings and model
relative_path = os.getcwd()

embeddig_path = relative_path + "/nlp/embeddings/"
model = 'glove-twitter-200' 
vectors = KeyedVectors.load(embeddig_path + model + '.bin')

model_path = relative_path + "/MoE_0/"+"yolov8n.pt"
model = YOLO(model_path)

async def run_MoE_0_entity(websocket, images, task):
    total = len(images)
    print(f"Tasks for MoE_0: {task}")
    bestScores = []

    for i, (image,prevScore) in enumerate(images):
        progress_value = 60 + (i / total) * 10
        response = {
            "message": f"Running 3/4 vision experts: Entity - {i}/{total}",
            "output": "nothing",
            "progress": f"{progress_value:.2f}"  
        }
        await websocket.send_text(json.dumps(response))
    
        print(f"\n")
           
        # Model predicions
        result = model(image, verbose=False)[0]  
        detections = []
        boxes = result.boxes

        # Unpackage predictions
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            detections.append({'class_name': class_name})

        # Count detected classes
        class_names = [d['class_name'] for d in detections]
        class_counts = Counter(class_names)

        print(f"Image {i} {images[i][0]} class counts:")
        for cls, count in class_counts.items():
            print(f"  > {cls}: {count}")

        # Scoring
        print(f"Scoring classes based on tasks {task}")
        score = 0
        for task_class, target_count in task.items():
            best_sim = 0
            for image_class, count in class_counts.items():
                print(f" > {image_class}: {count}")
                if task_class in vectors.key_to_index and image_class in vectors.key_to_index:
                    similarity = vectors.similarity(task_class, image_class)
                    print(f"    {task_class} ~ {image_class} = {similarity}")
                    best_sim = max(best_sim, similarity)
            
                score += best_sim + (count/target_count if best_sim>0.9 else 0)  + prevScore
            print(f" Image score: {score} with best_sim as {best_sim} updated from {task_class}")
        bestScores.append((image, score))

    bestScores.sort(key=lambda x: x[1], reverse=True)
    best_image_index = bestScores[0][0] if bestScores else None

    bestScores = bestScores[:10]
    print(f"\nBest scores updated (top 10):")
    print(bestScores)
    return bestScores