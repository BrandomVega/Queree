import requests
from PIL import Image
from io import BytesIO  
import random as rnd
import string
import os

def unique_name(formato, longitud=12):
    caracteres = string.ascii_letters + string.digits
    return ''.join(rnd.choices(caracteres, k=longitud)) + formato

width = [320, 480, 640, 768, 960, 1024, 1280, 1440, 1600]
height = [240, 360, 480, 576, 540, 768, 720, 900, 1000]

variations = len(width)
img_formats = ['.jpg', '.jpeg', '.png']
num_images = 100

for i in range(num_images):
    randomsize = rnd.randint(0,variations-1)
    w = width[randomsize]
    h = height[randomsize]
    w=200
    h=300
    img_format = img_formats[rnd.randint(0,len(img_formats)-1)]
    nombre = unique_name(img_format)
    print(f"Descargando {nombre}: {w}x{h}")
    
    r = requests.get(f'https://picsum.photos/{w}/{h}')
    im = Image.open(BytesIO(r.content))
    im.save(os.path.join("./target",nombre))
    print(f"Hecho")

    
