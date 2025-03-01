from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil

app = FastAPI()

model = YOLO("D:\\signature-detector-FastAPI\\best.pt")

TEMP_DIR = "./temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/detect-signature/")
async def detect_signature(file: UploadFile = File(...)):
    try:
        
        temp_file_path = os.path.join(TEMP_DIR, file.filename)
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        image = cv2.imread(temp_file_path)
        if image is None:
            return {"error": "Invalid image file or failed to read the image."}

        results = model(image)

        signature_list = []
        confidence_threshold = 0.7

        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = box.conf[0] 
                if conf < confidence_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cropped_signature = image[y1:y2, x1:x2]
                signature_list.append(cropped_signature)

        if not signature_list:
            return {"message": "No signatures detected."}
        else:
            widths = [sig.shape[1] for sig in signature_list]
            max_width = max(widths)
            
            padded_signatures = []
            for sig in signature_list:
                h, w = sig.shape[:2]
                if w < max_width:
                    pad_right = max_width - w
                    sig = cv2.copyMakeBorder(sig, 0, 0, 0, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                padded_signatures.append(sig)
            
            combined_image = cv2.vconcat(padded_signatures)
            
            output_image_path = os.path.join(TEMP_DIR, "detected.png")
            cv2.imwrite(output_image_path, combined_image)
            
            return {"message": "Detected signatures saved.", "output_path": output_image_path}
    
    except Exception as e:
        return {"error": str(e)}
