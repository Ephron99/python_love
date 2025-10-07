from PIL import Image
import pytesseract
import os
import json

# Set path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_image(image_path):
    """Extract full text from one image."""
    text = pytesseract.image_to_string(Image.open(image_path), lang="eng")
    return text.strip()

def main():
    folder = r"C:\Users\Public\Documents\Plustek-SecureScan\Image"
    results = []

    # Check if folder exists
    if not os.path.exists(folder):
        print("⚠️ Folder not found:", folder)
        return

    # Process all images
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
            path = os.path.join(folder, file)
            print(f"🖼️ Processing: {file} ...")
            try:
                text = extract_text_from_image(path)
                results.append({
                    "file": file,
                    "content": text
                })
            except Exception as e:
                results.append({
                    "file": file,
                    "error": str(e)
                })

    # Print JSON result
    json_result = json.dumps(results, indent=2, ensure_ascii=False)
    print("\n✅ Extracted JSON Output:")
    print(json_result)

    # Optionally save JSON to a file
    output_path = os.path.join(folder, "extracted_text.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_result)
    print(f"\n💾 Results saved to: {output_path}")

if __name__ == "__main__":
    main()
