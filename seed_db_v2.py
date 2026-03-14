import os
import shutil
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# Setup Groq for generation
# We will use Groq to quickly generate "Rich Descriptions" for 30 mock properties
from dotenv import load_dotenv
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

print("Initialize NexaSpace V2 Data Seeder...")
print("This will populate ChromaDB with 30 diverse properties (Indian, Gothic, Modern, Messy, Clean).")

# Get or create the local ChromaDB path
db_path = "./nexa_db"
# Delete old database to ensure fresh start with 30 items
if os.path.exists(db_path):
    print("Clearing old database 4-item database...")
    shutil.rmtree(db_path)
    
os.makedirs(db_path, exist_ok=True)

chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_or_create_collection(name="property_intents")

# 30 Highly Diverse Mock Properties focusing on specific global architectures and clutter levels
mock_properties = [
    # Indian Architecture
    {"id": "prop_01", "file_path": "https://images.unsplash.com/photo-1590074062493-2788fb6955a5?w=800&q=80", "style": "Traditional Indian Haveli", "flow": "Courtyard-centric open layout", "light": "Harsh sunlight filtered through Jali work", "clutter": "Low Clutter", "potential": "A heritage boutique hotel"},
    {"id": "prop_02", "file_path": "https://images.unsplash.com/photo-1512343879784-a960bf40e7f2?w=800&q=80", "style": "Modern Chettinad", "flow": "Large pillared halls with athangudi tiles", "light": "Warm ambient morning glow", "clutter": "Medium (Packed furniture)", "potential": "Large joint family residence"},
    {"id": "prop_03", "file_path": "https://plus.unsplash.com/premium_photo-1661877303180-19a028c21048?w=800&q=80", "style": "Mumbai Apartment", "flow": "Compact but highly optimized", "light": "Floor to ceiling balcony windows", "clutter": "High (Boxes and laundry)", "potential": "Modern urban minimalist pad"},
    {"id": "prop_04", "file_path": "https://images.unsplash.com/photo-1582268611958-ebfd161ef9cf?w=800&q=80", "style": "Kerala Backwater Villa", "flow": "Sloping roofs with veranda wrapping", "light": "Dappled light through palms", "clutter": "Low Clutter", "potential": "Serene retirement home"},
    {"id": "prop_05", "file_path": "https://images.unsplash.com/photo-1542617303-7aa7b38d38b6?w=800&q=80", "style": "South Indian Temple Vibe", "flow": "Linear progression to a central sanctum area", "light": "Dark and moody with oil lamps", "clutter": "Medium (Religious artifacts)", "potential": "Spiritual retreat center"},
    
    # Industrial / Urban
    {"id": "prop_06", "file_path": "https://images.unsplash.com/photo-1513694203232-719a280e022f?w=800&q=80", "style": "Industrial Loft", "flow": "High ceilings, raw exposed brick", "light": "Moody, warm artificial lighting", "clutter": "Extremely High (Messy tenant)", "potential": "A creative studio"},
    {"id": "prop_07", "file_path": "https://images.unsplash.com/photo-1505843513577-22bb7d21e455?w=800&q=80", "style": "Brutalist Concrete", "flow": "Heavy geometric volumes", "light": "Harsh directional sun shafts", "clutter": "Low Clutter", "potential": "Avant-garde gallery space"},
    {"id": "prop_08", "file_path": "https://images.unsplash.com/photo-1497366216548-37526070297c?w=800&q=80", "style": "Warehouse Conversion", "flow": "Massive unbroken floorplate", "light": "Industrial skylights", "clutter": "High (Pallets and equipment)", "potential": "Tech startup headquarters"},
    {"id": "prop_09", "file_path": "https://images.unsplash.com/photo-1554995207-c18c203602cb?w=800&q=80", "style": "Modernist Glass Box", "flow": "Boundaryless indoor/outdoor", "light": "360-degree blinding natural light", "clutter": "Low", "potential": "Luxury bachelor pad"},
    {"id": "prop_10", "file_path": "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=800&q=80", "style": "Corporate Office", "flow": "Cubicle heavy", "light": "Fluorescent overheads", "clutter": "Medium (Office supplies)", "potential": "Co-working space conversion"},

    # Victorian / Historic
    {"id": "prop_11", "file_path": "https://images.unsplash.com/photo-1588854337236-6889d631faa8?w=800&q=80", "style": "Victorian Gothic", "flow": "Segmented traditional parlor rooms", "light": "Dimly lit, relies on historic fixtures", "clutter": "High (Antique hoarding)", "potential": "Authentic restoration project"},
    {"id": "prop_12", "file_path": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800&q=80", "style": "Edwardian Estate", "flow": "Grand entrance hall leading to suites", "light": "Large bay windows", "clutter": "Low Clutter", "potential": "Luxury family estate"},
    {"id": "prop_13", "file_path": "https://images.unsplash.com/photo-1502005097973-ef5690414373?w=800&q=80", "style": "Tudor Revival", "flow": "Cozy asymmetrical rooms", "light": "Leaded glass window light", "clutter": "Medium (Heavy drapery)", "potential": "Storybook bed and breakfast"},
    {"id": "prop_14", "file_path": "https://images.unsplash.com/photo-1533779283484-8ad4940aa3a8?w=800&q=80", "style": "Colonial", "flow": "Strict bilateral symmetry", "light": "Evenly spaced double-hung windows", "clutter": "Low Clutter", "potential": "Stately mayoral residence"},
    {"id": "prop_15", "file_path": "https://images.unsplash.com/photo-1510798831971-661eb04b3739?w=800&q=80", "style": "French Chateau", "flow": "Enfilade room progression", "light": "Tall French doors with garden light", "clutter": "High (Construction materials)", "potential": "Vineyard chateau"},

    # Rural / Nature
    {"id": "prop_16", "file_path": "https://images.unsplash.com/photo-1600607687920-4e2a09cf159d?w=800&q=80", "style": "Modern Farmhouse", "flow": "Cozy but spacious kitchen-to-living", "light": "Bright morning sun exposure", "clutter": "Medium (Kids toys everywhere)", "potential": "Thanksgiving hosting house"},
    {"id": "prop_17", "file_path": "https://images.unsplash.com/photo-1449844908441-8829872d2607?w=800&q=80", "style": "A-Frame Cabin", "flow": "Vertical loft layout", "light": "Massive triangular window wall", "clutter": "Low Clutter", "potential": "Winter ski retreat"},
    {"id": "prop_18", "file_path": "https://images.unsplash.com/photo-1542314831-c6a4d14effc8?w=800&q=80", "style": "Log Cabin", "flow": "Central hearth focused", "light": "Dark wood absorbs light", "clutter": "High (Firewood and gear)", "potential": "Off-grid survival lodge"},
    {"id": "prop_19", "file_path": "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=800&q=80", "style": "Desert Adobe", "flow": "Thick walled passive cooling rooms", "light": "Tiny windows to block intense sun", "clutter": "Low Clutter", "potential": "Meditation retreat"},
    {"id": "prop_20", "file_path": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800&q=80", "style": "Treehouse", "flow": "Suspended organic layout", "light": "Canopy filtered light", "clutter": "Low Clutter", "potential": "Glamping Airbnb"},

    # Suburban / Standard
    {"id": "prop_21", "file_path": "https://images.unsplash.com/photo-1512917774080-9991f1c4c750?w=800&q=80", "style": "Suburban McMansion", "flow": "Oversized, disjointed rooms", "light": "Standard tract housing windows", "clutter": "Medium (Staged furniture)", "potential": "Large suburban family living"},
    {"id": "prop_22", "file_path": "https://images.unsplash.com/photo-1600566752355-35792bedcfea?w=800&q=80", "style": "Split-Level", "flow": "Half-stair segmentation", "light": "Decent front light, dark basement", "clutter": "High (Unpacked moving boxes)", "potential": "Starter home for young couple"},
    {"id": "prop_23", "file_path": "https://images.unsplash.com/photo-1564013799919-ab600027ffc6?w=800&q=80", "style": "Ranch Style", "flow": "Long single-story horizontal", "light": "Sliding glass back doors", "clutter": "Low Clutter", "potential": "Accessible elderly living"},
    {"id": "prop_24", "file_path": "https://images.unsplash.com/photo-1605276374104-aa237f7cecd7?w=800&q=80", "style": "Townhouse", "flow": "Narrow and vertical", "light": "Only front and back light", "clutter": "Medium", "potential": "Low maintenance urban living"},
    {"id": "prop_25", "file_path": "https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800&q=80", "style": "Craftsman Bungalow", "flow": "Built-in heavy, front porch focused", "light": "Overhanging eaves block high sun", "clutter": "High (Hoarder situation)", "potential": "Charming historical flip"},

    # Unique / Abstract
    {"id": "prop_26", "file_path": "https://images.unsplash.com/photo-1574362848149-11496d93a7c7?w=800&q=80", "style": "Mid-Century Modern", "flow": "Excellent open-concept layout", "light": "Floor-to-ceiling windows provide massive natural light", "clutter": "Medium Clutter (Boxes on floor)", "potential": "Perfect for hosting and entertaining guests"},
    {"id": "prop_27", "file_path": "https://images.unsplash.com/photo-1533090161767-e6ffed986c88?w=800&q=80", "style": "Minimalist Zen", "flow": "Negative space prioritized", "light": "Soft diffused shoji light", "clutter": "Zero Clutter", "potential": "Spa-like sanctuary"},
    {"id": "prop_28", "file_path": "https://images.unsplash.com/photo-1499955085172-a104c9463ece?w=800&q=80", "style": "Bohemian Maximalist", "flow": "Cramped but cozy nooks", "light": "Colorful filtered light through plants", "clutter": "Intentionally Extremely High", "potential": "Artist commune"},
    {"id": "prop_29", "file_path": "https://images.unsplash.com/photo-1505691938895-1758d7bef511?w=800&q=80", "style": "Cyberpunk Neon", "flow": "Tight cyber-cafe layout", "light": "Harsh neon pinks and blues", "clutter": "Extremely High (Wires and server racks)", "potential": "Underground hacker den"},
    {"id": "prop_30", "file_path": "https://images.unsplash.com/photo-1502672260266-1c1c24240f33?w=800&q=80", "style": "Art Deco", "flow": "Sleek metallic curves", "light": "Glamorous chandelier lighting", "clutter": "Low Clutter", "potential": "High-end Gatsby venue"}
]

documents = []
metadatas = []
ids = []

print("Generating rich vector descriptions via Groq...")
for i, prop in enumerate(mock_properties):
    # Create the vector description simply without hitting the API to save extreme amounts of time
    rich_desc = f"A {prop['style']} property with {prop['clutter']} level of clutter. The spatial flow features {prop['flow']}. Lighting is characterized by {prop['light']}. Overall, it possesses massive potential as a {prop['potential']}."
    
    documents.append(rich_desc)
    metadatas.append({
        "file_path": prop["file_path"],
        "architectural_style": prop["style"],
        "spatial_flow": prop["flow"],
        "lighting": prop["light"],
        "clutter_factor": prop["clutter"],
        "true_potential": prop["potential"]
    })
    ids.append(prop["id"])
    print(f"Processed {i+1}/30")

print("Adding items to ChromeDB...")
# Add to Vector DB (Chroma handles the text->embedding automatically)
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Successfully seeded {collection.count()} properties into the database.")
