import chromadb
import json

# Connect to the local vector DB created by ingest.py
chroma_client = chromadb.PersistentClient(path="./nexa_db")

try:
    collection = chroma_client.get_collection(name="property_intents")
except Exception as e:
    print("❌ Could not find the 'property_intents' database.")
    print("Make sure you run 'python ingest.py' first to build the vectors.")
    exit()

def semantic_search(query: str, num_results: int = 2):
    """
    Takes a natural language query and finds the property images whose 'Rich Descriptions'
    are the closest vector match, regardless of keywords.
    """
    print(f"\n🔍 Searching Intent: '{query}'")
    
    # Chroma handles embedding the query text and calculating the closest neighbors
    results = collection.query(
        query_texts=[query],
        n_results=num_results
    )

    if not results['documents'][0]:
        print("No matches found.")
        return

    print("\n" + "="*50)
    print("🏆 TOP MATCHES FOUND 🏆")
    print("="*50)

    # Loop over the closest matches
    for i in range(len(results['documents'][0])):
        file_id = results['ids'][0][i]
        score = results['distances'][0][i]
        metadata = results['metadatas'][0][i]
        
        print(f"\n🥇 MATCH #{i+1} | Distance: {score:.3f}")
        print(f"📷 Image Path: {metadata.get('file_path')}")
        print(f"🏛️ Style: {metadata.get('architectural_style')}")
        print(f"🌊 Flow: {metadata.get('spatial_flow')}")
        print(f"🗑️ Clutter: {metadata.get('clutter_factor')}")
        print(f"✨ Insight: {metadata.get('true_potential')}")
        print("-" * 50)

if __name__ == "__main__":
    print("\nWelcome to NexaSpace Intent Search!")
    print("Type 'exit' to quit.")
    
    while True:
        user_intent = input("\nDescribe your dream space (e.g. 'Industrial vibe but great lighting for hosting'): ")
        
        if user_intent.lower() == 'exit':
            break
            
        semantic_search(user_intent)
