import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import numpy as np
import Levenshtein
import json
import cv2
import re

def lev_distance(target, text):
    print(f"Calculating distance of found task and text")
    text = re.findall(r'\b[a-zA-Z0-9_]+\b', text)
    if len(text)>1:
        image_score = 0
        for looking in target.split():
            word_score = []
            for word in text:
                score = Levenshtein.distance(looking,word)
                word_score.append((word, score))
                #print(f"buscando {looking} with {word} = {Levenshtein.distance(looking,word)}")
            print(word_score)
            word_score.sort(key=lambda x: x[1])
            looking_best = word_score[0][1]
            image_score += looking_best
        print(f">>> Result for {target} = {image_score}")
        return image_score
    

def preprocessing(image):
    print(f"Preprocessing image: {image}")
    # Load the image
    image = cv2.imread(image)

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Apply threshold to the grayscale image
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(image, contours, -1, (240, 0, 159), 3)

    H, W = image.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
            break

    # Create a mask to isolate the text regions
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    text_region = cv2.bitwise_and(image, image, mask=mask)

    gray_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    gray_text_region = cv2.medianBlur(gray_text_region, 3)
    gray_text_region = cv2.threshold(gray_text_region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Morphological operations
    #kernel = np.ones((3, 3), np.uint8)
    #gray_text_region = cv2.erode(gray_text_region, kernel, iterations=1)

    scale_percent = 200  
    width = int(gray_text_region.shape[1] * scale_percent / 100)
    height = int(gray_text_region.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray_text_region, dim, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite('preprocessed_image.png', resized)
    return resized


def preprocessing_visualized(image_path):
    print(f"Preprocessing image: {image_path}")

    # Load the image
    image = cv2.imread(image_path)
    #cv2.imshow("Original ", image)
    #cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("Grayscale", gray)
    #cv2.waitKey(0)

    # Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imshow("Blurred", blurred)
    #cv2.waitKey(0)

    # Canny Edge Detection
    edged = cv2.Canny(blurred, 75, 200)
    #cv2.imshow("Edges", edged)
    #cv2.waitKey(0)

    # Thresholding
    thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("Thresholded", thresh)
    #cv2.waitKey(0)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contoured_image = image.copy()
    cv2.drawContours(contoured_image, contours, -1, (240, 0, 159), 3)
    #cv2.imshow("All Contours", contoured_image)
    #cv2.waitKey(0)

    # Filter for the central, square-ish contour
    H, W = image.shape[:2]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 100 and (0.7 < w/h < 1.3) and (W/4 < x + w//2 < W*3/4) and (H/4 < y + h//2 < H*3/4):
            break

    # Create mask and isolate the selected contour
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    #cv2.imshow("Mask", mask)
    #cv2.waitKey(0)

    text_region = cv2.bitwise_and(image, image, mask=mask)
    #cv2.imshow("Text region isolated", text_region)
    #cv2.waitKey(0)

    # Convert to grayscale and clean
    gray_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
    gray_text_region = cv2.medianBlur(gray_text_region, 3)
    #cv2.imshow("Cleaned Text region", gray_text_region)
    #cv2.waitKey(0)

    # Threshold again
    gray_text_region = cv2.threshold(
        gray_text_region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]
    #cv2.imshow("Final Thresholded", gray_text_region)
    #cv2.waitKey(0)

    # Resize
    scale_percent = 200
    width = int(gray_text_region.shape[1] * scale_percent / 100)
    height = int(gray_text_region.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray_text_region, dim, interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("Resized Output", resized)
    #cv2.waitKey(0)

    # Save final result
    cv2.imwrite("preprocessed_image.png", resized)
    #cv2.destroyAllWindows()

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    axes = axes.ravel()

    images = [
        (image, "Original"),
        (gray, "Grayscale"),
        (blurred, "Blurred"),
        (edged, "Canny Edges"),
        (thresh, "Thresholded"),
        (contoured_image, "Contours"),
        (mask, "Mask"),
        (gray_text_region, "Text region"),
        (resized, "Output"),
    ]

    for i, (img, title) in enumerate(images):
        if len(img.shape) == 2:  # Imagen en escala de grises
            axes[i].imshow(img, cmap='gray')
        else:
            axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

    return resized


def ocr_(resized):
    print(f"Preprocessing saved")
    print(f"Runing ocr engine")
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(resized, config=custom_config, lang='eng')
    print(text)
    print("Done")
    return text.lower()

#image = "img.jpg"
#ready = preprocessing_visualized(image)
#text = ocr_(ready)    

async def run_ocr(websocket, images, tasks):
    print(f"Iniciando OCR...")   
    total = len(images)
    
    for i,(image, prevScores) in enumerate(images):
        progress_value = 80 + (i / total) * 10
        response = {
            "message": f"Running 4/4 vision experts: OCR - {i}/{total}",
            "output": "nothing",
            "progress": f"{progress_value:.2f}"  
        }
        await websocket.send_text(json.dumps(response))

        ready = preprocessing(image)
        text = ocr_(ready)
        print(text)
        score = lev_distance(tasks[0], text)

        if score != None:
            if score == 0: score=1
            score = 1/score
        else:
            score = 0   
        print(f"Image score {image} = {score}")
        images[i] = list(images[i])        
        images[i][1] += score              # todo: add prev scores
        images[i] = tuple(images[i])      

    images.sort(key=lambda x: x[1], reverse=True)
    return images

#run_ocr([("bus.jpg",0)], 'cero emisiones abcd sevilla')