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

def detect_document_type(text):
    """Detect if the document is an ID card or passport."""
    text_upper = text.upper()
    
    # Check for passport indicators
    if any(keyword in text_upper for keyword in ["PASSPORT", "PASIPORO", "PASSEPORT", "PASIPOTI"]):
        return "passport"
    
    # Check for ID card indicators
    if any(keyword in text_upper for keyword in ["IDENTITY CARD", "INDANGAMUNTU", "NATIONAL ID"]):
        return "id_card"
    
    # Default to ID card if unsure
    return "id_card"

def extract_passport_names(text):
    """
    Extract names from passport.
    Prioritizes MRZ (last lines) as they are clearest.
    """
    lines = text.split('\n')
    surname = ""
    other_names = ""
    
    # METHOD 1 (PRIORITY): Look for MRZ lines (usually last 2 lines)
    # MRZ is most reliable as it's designed to be machine-readable
    # Check last 5 lines to be safe
    for line in reversed(lines[-5:]):
        line = line.strip()
        
        # Pattern 1: P<COUNTRY_CODE followed by name
        # Example: P<USAGUPTA<<RAHUL<RAM or PCRWANIYONZIMA<<CLAUDINE
        mrz_match = re.search(r'P[C<]?[A-Z]{3}([A-Z]+)<<([A-Z<]+)', line)
        if mrz_match:
            surname = mrz_match.group(1).replace('<', '').strip()
            other_names = mrz_match.group(2).replace('<', ' ').strip()
            return f"{surname} {other_names}".strip()
        
        # Pattern 2: Alternative MRZ format (starts with PC or P<)
        if line.startswith('PC') or line.startswith('P<'):
            # Extract everything after country code
            name_part = re.search(r'P[C<]?[A-Z]{3}([A-Z<]+)', line)
            if name_part:
                names_raw = name_part.group(1)
                # Split by << (surname separator from other names)
                if '<<' in names_raw:
                    parts = names_raw.split('<<')
                    surname = parts[0].replace('<', '').strip()
                    if len(parts) > 1:
                        other_names = parts[1].replace('<', ' ').strip()
                    return f"{surname} {other_names}".strip() if other_names else surname
    
    # METHOD 2 (FALLBACK): Look for surname and other names labels in full text
    for i, line in enumerate(lines):
        line_upper = line.upper()
        
        # Check for surname variations
        if any(keyword in line_upper for keyword in ["SURNAME", "NOM", "IRINA"]) and not surname:
            # Check next 2 lines for the actual name
            for j in range(1, 3):
                if i + j < len(lines):
                    words = lines[i + j].split()
                    name_regex = r'\b([A-Z]{2,})\b'  # At least 2 uppercase letters
                    for word in words:
                        matches = re.findall(name_regex, word)
                        if matches and len(matches[0]) >= 3:  # Avoid short words like "DE"
                            surname = matches[0]
                            break
                    if surname:
                        break
        
        # Check for other names/first names
        if any(keyword in line_upper for keyword in ["OTHER NAMES", "PRÉNOMS", "PRÉNOM", "ANDI MAZINA", "GIVEN"]) and not other_names:
            for j in range(1, 3):
                if i + j < len(lines):
                    words = lines[i + j].split()
                    name_regex = r'\b([A-Z]{2,})\b'
                    names_list = []
                    for word in words:
                        matches = re.findall(name_regex, word)
                        for match in matches:
                            if len(match) >= 3:
                                names_list.append(match)
                    if names_list:
                        other_names = " ".join(names_list)
                        break
                    if other_names:
                        break
    
    # Combine names
    if surname and other_names:
        return f"{surname} {other_names}"
    elif surname:
        return surname
    elif other_names:
        return other_names
    
    return ""

def extract_passport_id(text):
    """
    Extract passport number from text.
    Prioritizes MRZ lines (last lines) as they are clearest.
    """
    lines = text.split('\n')
    
    # METHOD 1 (PRIORITY): Extract from MRZ lines (last 5 lines)
    for line in reversed(lines[-5:]):
        line = line.strip()
        
        # Pattern 1: PC followed by 6 digits (Rwanda style)
        match = re.search(r'\b(PC\d{6})\b', line)
        if match:
            return match.group(1)
        
        # Pattern 2: 9 consecutive digits at start of line (US/common format)
        # MRZ second line format: passport_number<check_digit>country_code...
        match = re.match(r'^(\d{9})', line)
        if match:
            return match.group(1)
        
        # Pattern 3: Look for any 8-9 digit number in MRZ line
        match = re.search(r'\b(\d{8,9})\b', line)
        if match:
            return match.group(1)
    
    # METHOD 2 (FALLBACK): Search entire text
    # Rwanda passport format
    matches = re.findall(r'\b(PC\d{6})\b', text)
    if matches:
        return matches[0]
    
    # Look for passport number labels
    for i, line in enumerate(lines):
        line_upper = line.upper()
        if any(keyword in line_upper for keyword in ["PASSPORT NO", "N° PASSEPORT", "NUMERO", "NO."]):
            # Check current line and next 2 lines
            for j in range(0, 3):
                if i + j < len(lines):
                    check_line = lines[i + j]
                    
                    # PC format
                    match = re.search(r'(PC\d{6})', check_line)
                    if match:
                        return match.group(1)
                    
                    # 9 digit format
                    match = re.search(r'\b(\d{9})\b', check_line)
                    if match:
                        return match.group(1)
                    
                    # 8 digit format
                    match = re.search(r'\b(\d{8})\b', check_line)
                    if match:
                        return match.group(1)
    
    return ""

def extract_names_and_id(text):
    """
    Main function to extract names and ID based on document type.
    Returns: (names, id, document_type)
    """
    doc_type = detect_document_type(text)
    
    if doc_type == "passport":
        names = extract_passport_names(text)
        doc_id = extract_passport_id(text)
    else:  # id_card
        names = get_names(text)
        doc_id = get_id(text)
    
    return names, doc_id, doc_type

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
    Supports both ID cards and passports.
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
            
            # Extract names, ID, and detect document type
            names, doc_id, doc_type = extract_names_and_id(text)
            
            results.append({
                "filename": file,
                "document_type": doc_type,
                "names": names if names else None,
                "id": doc_id if doc_id else None,
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
    Extract names and ID from a single ID card or passport image.
    
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
        
        # Extract names, ID, and detect document type
        names, doc_id, doc_type = extract_names_and_id(text)
        
        return {
            "filename": file.filename,
            "document_type": doc_type,
            "preprocessing_method": method,
            "content": text,
            "extracted_names": names if names else None,
            "extracted_id": doc_id if doc_id else None,
            "success": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/extract-batch")
async def extract_from_multiple_images(files: List[UploadFile] = File(...)):
    """
    Extract names and ID from multiple ID card or passport images.
    
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
            
            # Extract names, ID, and detect document type
            names, doc_id, doc_type = extract_names_and_id(text)
            
            results.append({
                "filename": file.filename,
                "document_type": doc_type,
                "preprocessing_method": method,
                "content": text,
                "extracted_names": names if names else None,
                "extracted_id": doc_id if doc_id else None,
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