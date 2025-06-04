import numpy as np
import os

moe1_scene_classes = ['a','b','c']
moe2_obj_classes = ['u','v','w']
moe3_ocr = []

tasks = {
    "moe1":['a','c'],
    "moe2":['w','u'],
    "moe3":['cero emisiones']
}

print(f"best_scores = []")
for key in tasks:
    print(f"Tarea para {key}:")
    print(f"    INSIDE_{key.capitalize()}")
    print(f"    for image in target:\n\tresults = modelProcess()\n\ttarget = {tasks[key]}\n\tlocabest = ponderateResults(results,target)\n\tbest_scores.append(localbest)")