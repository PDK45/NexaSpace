import os
import chromadb
from sentence_transformers import SentenceTransformer

print("Initialize NexaSpace Dummy Data Seeder...")
print("This will populate ChromaDB with 4 impressive demo properties so the UI works immediately.")

# Get or create the local ChromaDB path
db_path = "./nexa_db"
os.makedirs(db_path, exist_ok=True)

# Important: We must use the same embedding model Chroma uses by default 
# so the dimensions match when the user searches in FastAPI via Chroma's built-in query.
chroma_client = chromadb.PersistentClient(path=db_path)

# Try to get the collection, or create it if missing
collection = chroma_client.get_or_create_collection(name="property_intents")

# Check if we already have data to avoid duplicating
if collection.count() > 0:
    print(f"Collection already has {collection.count()} items. Skipping seed. (Delete the nexa_db folder if you want to reset).")
    exit()

print("Collection is empty. Generating mock embeddings...")

# 4 Extremely High Quality Mock Properties
mock_properties = [
    {
        "id": "demo_prop_01",
        "file_path": "https://images.unsplash.com/photo-1574362848149-11496d93a7c7?w=800&q=80",
        "architectural_style": "Mid-Century Modern",
        "spatial_flow": "Excellent open-concept layout",
        "lighting": "Floor-to-ceiling windows provide massive natural light",
        "clutter_factor": "Medium Clutter (Boxes on floor)",
        "true_potential": "Perfect for hosting and entertaining guests",
        "rich_description": "An exceptional mid-century modern aesthetic with massive potential for natural light. Despite the current clutter, the foundational bones show an excellent open-concept layout that is perfect for hosting."
    },
    {
        "id": "demo_prop_02",
        "file_path": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=800&q=80",
        "architectural_style": "Industrial Loft",
        "spatial_flow": "High ceilings, raw exposed brick",
        "lighting": "Moody, warm artificial lighting",
        "clutter_factor": "Extremely High (Messy tenant)",
        "true_potential": "A creative studio or impressive urban dwelling",
        "rich_description": "A striking industrial loft vibe featuring high ceilings and raw exposed brick. The current tenant is incredibly messy, but the moody aesthetic is undeniable."
    },
    {
        "id": "demo_prop_03",
        "file_path": "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?w=800&q=80",
        "architectural_style": "Modern Farmhouse",
        "spatial_flow": "Cozy but spacious kitchen-to-living transition",
        "lighting": "Bright morning sun exposure",
        "clutter_factor": "Low Clutter",
        "true_potential": "A warm family home with a kitchen big enough to host Thanksgiving.",
        "rich_description": "A stunning modern farmhouse. It features a cozy but spacious transition from the living space into a kitchen that is big enough to host Thanksgiving with bright morning sun exposure."
    },
    {
        "id": "demo_prop_04",
        "file_path": "https://images.unsplash.com/photo-1588854337236-6889d631faa8?w=800&q=80",
        "architectural_style": "Victorian Gothic",
        "spatial_flow": "Segmented traditional parlor rooms",
        "lighting": "Dimly lit, relies on historic fixtures",
        "clutter_factor": "High ( Antique hoarding )",
        "true_potential": "A historically authentic restoration project",
        "rich_description": "A heavy victorian gothic style with lots of clutter. The segmented traditional rooms offer a deeply historic feel, despite the antique hoarding currently filling the space."
    }
]

# Extract data arrays for Chroma
documents = [prop["rich_description"] for prop in mock_properties]
ids = [prop["id"] for prop in mock_properties]
metadatas = [
    {
        "file_path": prop["file_path"],
        "architectural_style": prop["architectural_style"],
        "spatial_flow": prop["spatial_flow"],
        "lighting": prop["lighting"],
        "clutter_factor": prop["clutter_factor"],
        "true_potential": prop["true_potential"]
    }
    for prop in mock_properties
]

print("Adding items to ChromeDB...")
# Add to Vector DB (Chroma handles the text->embedding automatically)
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"✅ Successfully seeded {collection.count()} properties into the AI Search Engine.")
print("The frontend UI will now work perfectly!")
