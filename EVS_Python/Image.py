from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
import re
import os
from typing import List
import uvicorn

# Set path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default folder path
SCAN_FOLDER = r"C:\Users\Public\Documents\Plustek-SecureScan\Image"

app = FastAPI(title="OCR ID Card Extraction API", version="1.0.0")

# ============ Helper Functions ============

def remove_empty_lines(text):
    """Remove empty lines from text."""
    lines = text.split('\n')
    return '\n'.join([line for line in lines if line.strip()])

def remove_space(text):
    """Remove all whitespace from string."""
    return re.sub(r'\s', '', text)

def find_longest_integer(input_text):
    """Find the longest sequence of consecutive digits in the text."""
    longest_integer = ""
    current_integer = ""
    
    for char in input_text:
        if char.isdigit():
            current_integer += char
        else:
            if len(current_integer) > len(longest_integer):
                longest_integer = current_integer
            current_integer = ""
    
    if len(current_integer) > len(longest_integer):
        longest_integer = current_integer
    
    return longest_integer

def get_id(input_text):
    """Extract ID by removing spaces and finding the longest integer sequence."""
    text_no_space = remove_space(input_text)
    return find_longest_integer(text_no_space)

def get_names(input_text):
    """Extract names from text by finding the line 2 positions after 'CARD'."""
    input_text = remove_empty_lines(input_text)
    line_values = input_text.split('\n')
    fullnames = ""
    
    for i in range(len(line_values)):
        if "CARD" in line_values[i]:
            if i + 2 < len(line_values):
                target_line = line_values[i + 2]
                words = target_line.split()
                
                full_name_regex = r'\b([A-Z][a-zA-Z]*)\b'
                
                for word in words:
                    if word:
                        matches = re.findall(full_name_regex, word)
                        for name in matches:
                            name = name.strip()
                            if name:
                                if not fullnames:
                                    fullnames = name
                                else:
                                    fullnames += " " + name
                break
    
    return fullnames

def preprocess_image(image_array):
    """Apply preprocessing techniques to improve OCR accuracy."""
    preprocessed_images = {}
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # 1. Original grayscale
    preprocessed_images['grayscale'] = gray
    
    # 2. Otsu's thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images['otsu'] = thresh_otsu
    
    # 3. Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    preprocessed_images['adaptive'] = adaptive_thresh
    
    # 4. Denoising + Otsu
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    _, denoised_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images['denoised'] = denoised_otsu
    
    # 5. Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    preprocessed_images['contrast'] = contrast_enhanced
    
    return preprocessed_images

def extract_text_with_preprocessing(image_array):
    """Extract text using multiple preprocessing methods and return the best result."""
    results = {}
    
    # Extract text from original image
    try:
        original_img = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        original_text = pytesseract.image_to_string(original_img, lang="eng").strip()
        results['original'] = {
            'text': original_text,
            'length': len(original_text)
        }
    except Exception as e:
        results['original'] = {
            'text': '',
            'length': 0
        }
    
    # Get preprocessed images and extract text from each
    try:
        preprocessed_images = preprocess_image(image_array)
        
        for method, processed_img in preprocessed_images.items():
            pil_img = Image.fromarray(processed_img)
            text = pytesseract.image_to_string(pil_img, lang="eng").strip()
            results[method] = {
                'text': text,
                'length': len(text)
            }
    except Exception as e:
        pass
    
    # Find the best result
    best_method = max(results.items(), key=lambda x: x[1]['length'])
    
    return {
        'best_method': best_method[0],
        'best_text': best_method[1]['text']
    }

# ============ API Endpoints ============

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "OCR ID Card Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/scan-folder": "GET - Scan images from default folder and extract data",
            "/extract": "POST - Extract text, names, and ID from a single image",
            "/extract-batch": "POST - Extract from multiple images"
        }
    }

@app.get("/scan-folder")
async def scan_folder():
    """
    Automatically scan all images in the default folder and extract names and IDs.
    Returns only the extracted names and IDs (not full content).
    """
    # Check if folder exists
    if not os.path.exists(SCAN_FOLDER):
        raise HTTPException(status_code=404, detail=f"Folder not found: {SCAN_FOLDER}")
    
    # Get all image files
    image_files = [f for f in os.listdir(SCAN_FOLDER) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp"))]
    
    if not image_files:
        return {
            "message": "No images found in folder",
            "folder": SCAN_FOLDER,
            "results": []
        }
    
    results = []
    
    for file in image_files:
        path = os.path.join(SCAN_FOLDER, file)
        
        try:
            # Read image
            img = cv2.imread(path)
            
            if img is None:
                results.append({
                    "filename": file,
                    "error": "Could not read image",
                    "success": False
                })
                continue
            
            # Extract text with preprocessing
            extraction_result = extract_text_with_preprocessing(img)
            text = extraction_result['best_text']
            method = extraction_result['best_method']
            
            # Extract names and ID
            extracted_names = get_names(text)
            extracted_id = get_id(text)
            
            results.append({
                "filename": file,
                "names": extracted_names if extracted_names else None,
                "id": extracted_id if extracted_id else None,
                "preprocessing_method": method,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "filename": file,
                "error": str(e),
                "success": False
            })
    
    return {
        "folder": SCAN_FOLDER,
        "total_images": len(image_files),
        "successful": len([r for r in results if r.get('success')]),
        "failed": len([r for r in results if not r.get('success')]),
        "results": results
    }

@app.post("/extract")
async def extract_from_image(file: UploadFile = File(...)):
    """
    Extract names and ID from a single ID card image.
    
    - **file**: Image file (jpg, jpeg, png, tif, bmp)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Extract text with preprocessing
        extraction_result = extract_text_with_preprocessing(img)
        text = extraction_result['best_text']
        method = extraction_result['best_method']
        
        # Extract names and ID
        extracted_names = get_names(text)
        extracted_id = get_id(text)
        
        return {
            "filename": file.filename,
            "preprocessing_method": method,
            "content": text,
            "extracted_names": extracted_names if extracted_names else None,
            "extracted_id": extracted_id if extracted_id else None,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract-batch")
async def extract_from_multiple_images(files: List[UploadFile] = File(...)):
    """
    Extract names and ID from multiple ID card images.
    
    - **files**: Multiple image files
    """
    results = []
    
    for file in files:
        # Validate file type
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "error": "File must be an image",
                "success": False
            })
            continue
        
        try:
            # Read image file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file",
                    "success": False
                })
                continue
            
            # Extract text with preprocessing
            extraction_result = extract_text_with_preprocessing(img)
            text = extraction_result['best_text']
            method = extraction_result['best_method']
            
            # Extract names and ID
            extracted_names = get_names(text)
            extracted_id = get_id(text)
            
            results.append({
                "filename": file.filename,
                "preprocessing_method": method,
                "content": text,
                "extracted_names": extracted_names if extracted_names else None,
                "extracted_id": extracted_id if extracted_id else None,
                "success": True
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
                "success": False
            })
    
    return {
        "total_files": len(files),
        "successful": len([r for r in results if r.get('success')]),
        "failed": len([r for r in results if not r.get('success')]),
        "results": results
    }

# ============ Run Server ============

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)