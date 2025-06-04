from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
import asyncio
import json
import time
import sys
import os

sys.path.append(os.path.abspath("./nlp/"))
sys.path.append(os.path.abspath("./MoE_0/"))
sys.path.append(os.path.abspath("./MoE_1/"))
sys.path.append(os.path.abspath("./MoE_2/"))
sys.path.append(os.path.abspath("./MoE_3/"))

from mobilenet_obj import run_MoE_1_imagenet
from mobilenet_places import run_MoE_2_places
from prc_gensim import run_nlp_taks
from process_input import p_input
from ocr import run_ocr
from yolo_ent import run_MoE_0_entity

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def find_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

# Send best scores to desktop-app
async def send_matching_images(websocket, best_matches):
    results = [{"image_path": f"file://{match[0]}", "similarity": match[1]} for match in best_matches]
    response = {
        "message": " Completed! click there to see the results",
        "output": results,  
        "progress": "100"
    }
    await websocket.send_text(json.dumps(response))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    # WebSocket connection from the electron client
    print(f"===========================")
    print(f"STARTING WEBSOCKET AT 8000")
    print(f"Sin conexion")
    await websocket.accept()
    print(f"Connected!")
    # Todo: verificar la conexion dentro de cada operacion para poder detener el procesamiento pq sino se sigue
    try:
        while True:
            # waits for client to start processing
            data = await websocket.receive_text()             
            client_data = json.loads(data)                    

            query = client_data['query']
            path = client_data['directory']

            # Starts processing
            # Manage image paths
            response = {
                "message": f"Running backend...",
                "progress":"0",
                "output":""
            }
            await websocket.send_text(json.dumps(response))
            time.sleep(1)
            target_imagepth = find_image_files(str(path)) 
            print(f"Total de imagenes en objetivo: {len(target_imagepth)}")
            response = {
                "message": f"Processing {len(target_imagepth)} images",
                "output":"", 
                "progress":"4"}
            await websocket.send_text(json.dumps(response))
            bestScores = [(path, 0) for path in target_imagepth]
            time.sleep(2)
    
            # NLP Module
            print(f"======"*20)
            response = {"message": f"Processing NL user input...","output":"", "progress":"8"}
            await websocket.send_text(json.dumps(response))

            tasks = p_input(query)

            response = {"message": f"NLP task completed", "output": str(tasks), "progress":"10"}
            await websocket.send_text(json.dumps(response))

            print(f"==="*10)
            print(f"VISION MODULE")
            print(f"==="*10)

            # Starting MoE_1: mobilenet objects        
            print(f"======"*20)
            response = {"message": f"Running 1/4 vision experts: Objects", "output": str(tasks), "progress":"20"}
            await websocket.send_text(json.dumps(response))            
            
            print(f"Enviando a MoE_1: Objects")
            if tasks.get('MoE_1'):  
                bestScores = await run_MoE_1_imagenet(websocket, bestScores, tasks['MoE_1'])
            else:
                print(f"Nothing to do here")

            # Starting MoE_2: mobilenet scenes        
            print(f"======"*20)
            response = {"message": f"Running 2/4 vision experts: Scenes", "output": str(tasks), "progress":"40"}
            await websocket.send_text(json.dumps(response))            
            
            print(f"Enviando a MoE_2: Scenes")
            if tasks.get('MoE_2'):  
                bestScores = await run_MoE_2_places(websocket, bestScores, tasks['MoE_2'])
            else:
                print(f"Nothing to do here")

            # Starting MoE_0: YOLO11 entities        
            print(f"======"*20)
            response = {"message": f"Running 3/4 vision experts: Entity", "output": str(tasks), "progress":"60"}
            await websocket.send_text(json.dumps(response))            
            bestScores = bestScores[:10]
            print(f"Enviando a MoE_0: Entidades.")
            if tasks.get('MoE_0'):  
                bestScores = await run_MoE_0_entity(websocket, bestScores, tasks['MoE_0'])
            else:
                print(f"Nothing to do here")

            # Starting MoE_3: OCR        
            print(f"======"*20)
            response = {"message": f"Running 4/4 vision experts: OCR", "output": str(tasks), "progress":"80"}
            await websocket.send_text(json.dumps(response))            
            
            print(f"Enviando a MoE_3: OCR.")
            if tasks.get('MoE_3'):  
                bestScores = await run_ocr(websocket, bestScores, tasks['MoE_3'])
            else:
                print(f"Nothing to do here")

            response = {"message": f"Completed! click there to see results", "output": str(tasks), "progress":"100"}
            await websocket.send_text(json.dumps(response))

            bestScores = bestScores[:10]
            await send_matching_images(websocket, bestScores)

            time.sleep(2)

    except WebSocketDisconnect:
        print("Client disconnected")

print(f"Backend Ready!")
# docker system prune
# docker run -d --name queree_app -p 8000:8000 -v /home/bran/:/home/bran/  queree_ready
# docker run -e PYTHONUNBUFFERED=1 my-python-app