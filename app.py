import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb

# Initialize FastAPI
app = FastAPI(title="NexaSpace Intent Search API")

# Allow CORS for frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to the local vector DB created by ingest.py
chroma_client = chromadb.PersistentClient(path="./nexa_db")

try:
    collection = chroma_client.get_collection(name="property_intents")
except Exception as e:
    print("Warning: Collection 'property_intents' not found. Run ingest.py first.")
    collection = None

# Set up Groq AI for lightning-fast dynamic reasoning on search results
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

groq_client = None
if groq_key:
    groq_client = Groq(api_key=groq_key)
    print("Active AI Reasoning powered by Groq (Llama 3) is Online.")
else:
    print("No GROQ_API_KEY found. Running in vector-only mode.")

# Pydantic model for incoming search queries
class SearchQuery(BaseModel):
    intent: str
    num_results: int = 4

@app.post("/api/search")
def search_properties(query: SearchQuery):
    if not collection:
        raise HTTPException(status_code=500, detail="Database not initialized. Please run ingestion first.")

    try:
        # Chroma handles embedding the query text and calculating the closest neighbors
        results = collection.query(
            query_texts=[query.intent],
            n_results=query.num_results
        )

        if not results['documents'][0]:
            return {"matches": []}

        # Limit to the top 2 absolute best matches visually so the screen isn't overwhelmed with all properties
        top_n = min(2, len(results['documents'][0]))
        
        formatted_results = []
        for i in range(top_n):
            metadata = results['metadatas'][0][i]
            
            # Check if file_path is external or local
            file_path = metadata.get('file_path', '')
            if file_path.startswith('http'):
                image_url = file_path
            else:
                # Assuming local static file
                image_url = f"/static/{file_path.replace(chr(92), '/')}" 

            # 🧠 Active AI Reasoning (The Magic UI feature via Groq)
            ai_reasoning = ""
            if groq_client and i == 0:  # Only reason for the top absolute match to save time
                try:
                    reason_prompt = f"A user searched for: '{query.intent}'. We found a property matching this description: '{results['documents'][0][i]}'. In one short, punchy, impressive sentence, explain why this property is the perfect match for their intent."
                    
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are NexaSpace, an elite AI real estate assistant. Respond with exactly one sentence."
                            },
                            {
                                "role": "user",
                                "content": reason_prompt,
                            }
                        ],
                        model="llama3-70b-8192",
                    )
                    ai_reasoning = chat_completion.choices[0].message.content.strip()
                except Exception as e:
                    print(f"Groq API Error: {e}")
                    pass

            formatted_results.append({
                "id": results['ids'][0][i],
                "score": float(results['distances'][0][i]),
                "image_url": image_url,
                "architectural_style": metadata.get('architectural_style', 'Unknown'),
                "spatial_flow": metadata.get('spatial_flow', 'Unknown'),
                "lighting": metadata.get('lighting', 'Unknown'),
                "clutter_factor": metadata.get('clutter_factor', 'Unknown'),
                "true_potential": metadata.get('true_potential', 'Unknown'),
                "rich_description": results['documents'][0][i],
                "ai_reasoning": ai_reasoning
            })

        return {"matches": formatted_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files to serve the frontend UI and the images
# Mounting the root dir allows us to serve the index.html and any local dataset images
app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
