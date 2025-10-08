from PIL import Image
import pytesseract
import cv2
import numpy as np
import os
import json
import re

# Set path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def remove_empty_lines(text):
    """Remove empty lines from text."""
    lines = text.split('\n')
    return '\n'.join([line for line in lines if line.strip()])

def remove_space(text):
    """Remove all whitespace from string."""
    return re.sub(r'\s', '', text)

def find_longest_integer(input_text):
    """
    Find the longest sequence of consecutive digits in the text.
    This is typically the ID number.
    """
    longest_integer = ""
    current_integer = ""
    
    for char in input_text:
        if char.isdigit():
            current_integer += char
        else:
            if len(current_integer) > len(longest_integer):
                longest_integer = current_integer
            current_integer = ""
    
    # Check the last sequence
    if len(current_integer) > len(longest_integer):
        longest_integer = current_integer
    
    return longest_integer

def get_id(input_text):
    """
    Extract ID by removing spaces and finding the longest integer sequence.
    """
    text_no_space = remove_space(input_text)
    return find_longest_integer(text_no_space)

def preprocess_image(image_path):
    """
    Apply preprocessing techniques to improve OCR accuracy.
    Returns multiple preprocessed versions of the image.
    """
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    preprocessed_images = {}
    
    # 1. Original grayscale
    preprocessed_images['grayscale'] = gray
    
    # 2. Otsu's thresholding (automatic threshold calculation)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images['otsu'] = thresh_otsu
    
    # 3. Adaptive thresholding (good for varying lighting)
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

def extract_text_with_preprocessing(image_path):
    """
    Extract text using multiple preprocessing methods and return the best result.
    """
    results = {}
    
    # Extract text from original image
    try:
        original_img = Image.open(image_path)
        original_text = pytesseract.image_to_string(original_img, lang="eng").strip()
        results['original'] = {
            'text': original_text,
            'length': len(original_text)
        }
    except Exception as e:
        results['original'] = {
            'text': '',
            'length': 0,
            'error': str(e)
        }
    
    # Get preprocessed images and extract text from each
    try:
        preprocessed_images = preprocess_image(image_path)
        
        for method, processed_img in preprocessed_images.items():
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(processed_img)
            text = pytesseract.image_to_string(pil_img, lang="eng").strip()
            results[method] = {
                'text': text,
                'length': len(text)
            }
    except Exception as e:
        print(f"     Preprocessing error: {str(e)}")
    
    # Find the best result (longest text, usually indicates better recognition)
    best_method = max(results.items(), key=lambda x: x[1]['length'])
    
    return {
        'best_method': best_method[0],
        'best_text': best_method[1]['text'],
        'all_results': {k: v['length'] for k, v in results.items()}
    }

def get_names(input_text):
    """
    Extract names from text by finding the line 2 positions after 'CARD'.
    Extracts all words that start with an uppercase letter (capitalized words).
    """
    input_text = remove_empty_lines(input_text)
    line_values = input_text.split('\n')
    fullnames = ""
    
    for i in range(len(line_values)):
        if "CARD" in line_values[i]:
            # Check if there's a line 2 positions after current line
            if i + 2 < len(line_values):
                target_line = line_values[i + 2]
                words = target_line.split()
                
                # Pattern to match words that start with uppercase letter
                # Matches: DEI, Francois, Audace, JOHN, etc.
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

def extract_text_from_image(image_path):
    """Extract text from image using the best preprocessing method."""
    extraction_result = extract_text_with_preprocessing(image_path)
    return extraction_result['best_text'], extraction_result['best_method']

def main():
    folder = r"C:\Users\Public\Documents\Plustek-SecureScan\Image"
    results = []
    
    # Check if folder exists
    if not os.path.exists(folder):
        print(" Folder not found:", folder)
        return
    
    # Process all images
    image_files = [f for f in os.listdir(folder) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp"))]
    
    print(f" Found {len(image_files)} images to process\n")
    
    for idx, file in enumerate(image_files, 1):
        path = os.path.join(folder, file)
        print(f" Processing [{idx}/{len(image_files)}]: {file} ...")
        
        try:
            text, method = extract_text_from_image(path)
            extracted_names = get_names(text)
            extracted_id = get_id(text)
            
            results.append({
                "file": file,
                "preprocessing_method": method,
                "content": text,
                "extracted_names": extracted_names if extracted_names else "No names found",
                "extracted_id": extracted_id if extracted_id else "No ID found"
            })
            
            print(f" Completed: {file} (Method: {method})")
            if extracted_names:
                print(f"   Names: {extracted_names}")
            if extracted_id:
                print(f"   ID: {extracted_id}")
            if not extracted_names and not extracted_id:
                print(f"    No names or ID found")
                
        except Exception as e:
            results.append({
                "file": file,
                "error": str(e)
            })
            print(f" Error: {file} - {str(e)}")
    
    # Print JSON result
    json_result = json.dumps(results, indent=2, ensure_ascii=False)
    print("\n" + "="*50)
    print(" Extracted JSON Output:")
    print("="*50)
    print(json_result)
    
    # Save JSON to a file
    output_path = os.path.join(folder, "extracted_text.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_result)
    
    print(f"\n Results saved to: {output_path}")
    
    # Print summary
    successful = len([r for r in results if 'content' in r])
    failed = len([r for r in results if 'error' in r])
    names_found = len([r for r in results if 'extracted_names' in r and r['extracted_names'] != "No names found"])
    ids_found = len([r for r in results if 'extracted_id' in r and r['extracted_id'] != "No ID found"])
    
    print(f"\n Summary:")
    print(f"   - Total processed: {len(results)}")
    print(f"   - Successful: {successful}")
    print(f"   - Failed: {failed}")
    print(f"   - Names extracted: {names_found}")
    print(f"   - IDs extracted: {ids_found}")

if __name__ == "__main__":
    main()