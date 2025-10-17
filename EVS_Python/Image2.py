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
import psycopg2
from psycopg2 import pool
from datetime import datetime

# Set path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Default folder path
SCAN_FOLDER = r"C:\Users\Public\Documents\Plustek-SecureScan\Image"

# Database configuration
DB_CONFIG = {
    "host": "localhost",
    "database": "evsdb",
    "user": "postgres",
    "password": "5432",  # UPDATE THIS
    "port": 5432
}

app = FastAPI(title="OCR ID Card Extraction API", version="1.0.0")

# Database connection pool
connection_pool = None

def init_db_pool():
    """Initialize database connection pool"""
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            host=DB_CONFIG["host"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            port=DB_CONFIG["port"]
        )
        print("✅ Database connection pool created successfully")
    except Exception as e:
        print(f"❌ Error creating connection pool: {e}")

def get_db_connection():
    """Get a connection from the pool"""
    if connection_pool:
        return connection_pool.getconn()
    return None

def release_db_connection(conn):
    """Return connection to the pool"""
    if connection_pool and conn:
        connection_pool.putconn(conn)

def create_attendance_table():
    """Create attendance table if it doesn't exist"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            print("❌ Could not get database connection")
            return
        
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            filename BYTEA NOT NULL,
            document_type VARCHAR(50) NOT NULL,
            names VARCHAR(255),
            doc_id VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        print("✅ Attendance table ready")
        
    except Exception as e:
        print(f"❌ Error creating table: {e}")
    finally:
        if conn:
            release_db_connection(conn)

def save_to_database(filename_blob, document_type, names, doc_id):
    """Save extracted data to database"""
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise Exception("Could not get database connection")
        
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO attendance (filename, document_type, names, doc_id)
        VALUES (%s, %s, %s, %s)
        RETURNING id;
        """
        
        cursor.execute(insert_query, (
            psycopg2.Binary(filename_blob),
            document_type,
            names,
            doc_id
        ))
        
        record_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        
        return record_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Database error: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db_pool()
    create_attendance_table()

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown"""
    global connection_pool
    if connection_pool:
        connection_pool.closeall()
        print("✅ Database connections closed")

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
    Extract names from passport MRZ (Machine Readable Zone).
    Based on the exact Java logic for precise MRZ parsing.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    for line in lines:
        if '<' in line:  # Find the first line with '<'
            line = line.strip()
            
            # Check if line is long enough
            if len(line) < 7:
                continue
            
            # Check first two characters
            first_two_contain_symbol = '<' in line[:2]
            
            if not first_two_contain_symbol:
                # Case 1: First two characters do not contain '<'
                # Example: PCRWANIYONZIMA<<CLAUDINE
                first_name_start = 5  # First name starts at 6th character (index 5)
                
                first_separator = line.find('<<', first_name_start)
                if first_separator == -1:
                    continue
                
                first_name = line[first_name_start:first_separator]
                
                second_separator = line.find('<', first_separator + 2)
                if second_separator == -1:
                    # If no further '<', take the rest
                    second_name = line[first_separator + 2:]
                else:
                    second_name = line[first_separator + 2:second_separator]
                
                return first_name.replace('<', '') + " " + second_name.replace('<', '')
            
            else:
                # Case 2: First two characters contain '<'
                # Example: P<USAGUPTA<<RAHUL<RAM
                first_separator = line.find('<<')
                if first_separator == -1:
                    continue
                
                second_separator = line.find('<', first_separator + 2)
                if second_separator == -1:
                    continue
                
                first_name = line[first_separator + 2:second_separator]
                
                third_separator = line.find('<', second_separator + 1)
                if third_separator == -1:
                    # Take the rest
                    second_name = line[second_separator + 1:]
                else:
                    second_name = line[second_separator + 1:third_separator]
                
                return first_name.replace('<', '') + " " + second_name.replace('<', '')
    
    return ""  # Return empty if no valid name found

def extract_passport_id(text):
    """
    Extract passport ID from MRZ second line.
    Based on the exact Java logic - extracts first 8 characters from second line.
    """
    if not text:
        return ""
    
    lines = text.split('\n')
    
    # Find the MRZ lines (lines containing '<')
    mrz_lines = [line.strip() for line in lines if '<' in line]
    
    # Ensure there is a second MRZ line
    if len(mrz_lines) < 2:
        return ""
    
    # Extract the first 8 characters from the second MRZ line
    second_line = mrz_lines[1]
    return second_line[:8] if len(second_line) >= 8 else second_line

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
            "/scan-and-save": "GET - Scan images and save to database",
            "/extract": "POST - Extract text, names, and ID from a single image",
            "/extract-and-save": "POST - Extract and save to database",
            "/extract-batch": "POST - Extract from multiple images",
            "/get-attendance": "GET - Get all attendance records"
        }
    }

@app.get("/scan-and-save")
async def scan_and_save():
    """
    Scan all images in the default folder, extract data, and save to database.
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
    saved_count = 0
    
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
            
            # Read image file as blob
            with open(path, 'rb') as f:
                image_blob = f.read()
            
            # Extract text with preprocessing
            extraction_result = extract_text_with_preprocessing(img)
            text = extraction_result['best_text']
            method = extraction_result['best_method']
            
            # Extract names, ID, and detect document type
            names, doc_id, doc_type = extract_names_and_id(text)
            
            # Save to database
            try:
                record_id = save_to_database(image_blob, doc_type, names, doc_id)
                
                results.append({
                    "filename": file,
                    "document_type": doc_type,
                    "names": names if names else None,
                    "id": doc_id if doc_id else None,
                    "database_id": record_id,
                    "saved": True,
                    "success": True
                })
                saved_count += 1
                
            except Exception as db_error:
                results.append({
                    "filename": file,
                    "document_type": doc_type,
                    "names": names if names else None,
                    "id": doc_id if doc_id else None,
                    "saved": False,
                    "database_error": str(db_error),
                    "success": False
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
        "successful_extractions": len([r for r in results if r.get('success')]),
        "saved_to_database": saved_count,
        "failed": len([r for r in results if not r.get('success')]),
        "results": results
    }

@app.post("/extract-and-save")
async def extract_and_save(file: UploadFile = File(...)):
    """
    Extract names and ID from a single ID card or passport image and save to database.
    
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
        
        # Save to database
        try:
            record_id = save_to_database(contents, doc_type, names, doc_id)
            
            return {
                "filename": file.filename,
                "document_type": doc_type,
                "preprocessing_method": method,
                "extracted_names": names if names else None,
                "extracted_id": doc_id if doc_id else None,
                "database_id": record_id,
                "saved_to_database": True,
                "success": True
            }
            
        except Exception as db_error:
            raise HTTPException(status_code=500, detail=f"Database error: {str(db_error)}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/get-attendance")
async def get_attendance():
    """
    Retrieve all attendance records from the database.
    """
    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            raise HTTPException(status_code=500, detail="Could not connect to database")
        
        cursor = conn.cursor()
        
        select_query = """
        SELECT id, document_type, names, doc_id, created_at
        FROM attendance
        ORDER BY created_at DESC;
        """
        
        cursor.execute(select_query)
        records = cursor.fetchall()
        cursor.close()
        
        attendance_list = []
        for record in records:
            attendance_list.append({
                "id": record[0],
                "document_type": record[1],
                "names": record[2],
                "doc_id": record[3],
                "created_at": record[4].isoformat() if record[4] else None
            })
        
        return {
            "total_records": len(attendance_list),
            "attendance": attendance_list
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving data: {str(e)}")
    finally:
        if conn:
            release_db_connection(conn)
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