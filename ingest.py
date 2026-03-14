import os
import json
import chromadb
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file.")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Configure Local ChromaDB Vector Store
chroma_client = chromadb.PersistentClient(path="./nexa_db")
collection = chroma_client.get_or_create_collection(name="property_intents")

def analyze_image(image_path: str):
    """"
    Uses Gemini Vision to perform deep spatial reasoning on an image.
    We instruct it to ignore clutter and focus on architectural potential, returning JSON.
    """
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

    prompt = """
    Act as an expert architectural and spatial analyst for a high-end real estate firm.
    Analyze this room photograph. Look past any temporary messes, clutter, or bad furniture.
    Focus on the "bones" of the room: the architectural style, spatial flow, natural lighting, and its true potential.

    Output ONLY a valid JSON object with the following keys:
    {
      "architectural_style": "String (e.g. Victorian, Industrial, Mid-Century Modern)",
      "spatial_flow": "String describing the layout and openness",
      "lighting": "String describing natural or artificial lighting quality",
      "clutter_factor": "String (e.g. High/Medium/Low - briefly explain why)",
      "true_potential": "String describing what the room could be used for",
      "rich_description": "A comprehensive 3-5 sentence summary combining the above that captures the emotional and spatial intent of the room. This will be used for semantic search."
    }
    """
    
    try:
        print(f"🧠 Reasoning about {os.path.basename(image_path)}...")
        response = model.generate_content([prompt, img])
        
        # Clean the response to ensure it's parseable JSON
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
        
        metadata = json.loads(text)
        return metadata

    except Exception as e:
        print(f"❌ Failed to analyze {image_path}: {e}")
        return None

def ingest_directory(directory_path: str):
    """
    Loops over images in a directory, analyzes them, and inserts them into ChromaDB.
    """
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} not found.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"No valid images found in {directory_path}.")
        return

    print(f"🚀 Found {len(files)} images. Starting ingestion pipeline...\n")

    for file in files:
        file_path = os.path.join(directory_path, file)
        
        # 1. Image -> Gemini Reasoner -> JSON
        analysis = analyze_image(file_path)
        
        if analysis:
            # 2. Extract the payload
            rich_description = analysis.get("rich_description", "")
            
            if not rich_description:
                print("Skipping due to missing rich_description.")
                continue

            # In ChromaDB, the "document" is what gets embedded into a vector
            # The "metadata" is payload we get back when a search matches
            
            # Format metadata so Chroma can store it easily (dict of strings)
            stored_metadata = {
                "file_path": file_path,
                "architectural_style": str(analysis.get('architectural_style', '')),
                "spatial_flow": str(analysis.get('spatial_flow', '')),
                "lighting": str(analysis.get('lighting', '')),
                "clutter_factor": str(analysis.get('clutter_factor', '')),
                "true_potential": str(analysis.get('true_potential', ''))
            }

            print(f"💾 Vectorizing and Storing '{os.path.basename(file_path)}'...")
            
            # 3. Add to ChromaDB
            # Chroma automatically handles the text-to-vector embedding using its default embedding model
            collection.add(
                documents=[rich_description],
                metadatas=[stored_metadata],
                ids=[file] # Using filename as unique ID
            )
            print("✅ Success!\n")

if __name__ == "__main__":
    # Ensure a 'sample_images' or 'datasets' folder exists
    # Change 'sample_images' to whatever folder you ran the download_datasets.py script into
    target_folder = "datasets/chaos"
    
    if not os.path.exists(target_folder):
        print(f"Creating a '{target_folder}' folder for you to place test images.")
        os.makedirs(target_folder, exist_ok=True)
        print("Please place a few messy room photos in there and run this script again.")
    else:
        ingest_directory(target_folder)
        print("🎉 Ingestion complete. Run search.py next to test the magic.")
