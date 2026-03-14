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

        formatted_results = []
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            # Ensure file path is accessible by the static server (assuming images are in the root directory relative to the script)
            # The static server will mount the current directory at /static
            image_url = f"/static/{metadata.get('file_path').replace(chr(92), '/')}" 

            formatted_results.append({
                "id": results['ids'][0][i],
                "score": float(results['distances'][0][i]),
                "image_url": image_url,
                "architectural_style": metadata.get('architectural_style', 'Unknown'),
                "spatial_flow": metadata.get('spatial_flow', 'Unknown'),
                "lighting": metadata.get('lighting', 'Unknown'),
                "clutter_factor": metadata.get('clutter_factor', 'Unknown'),
                "true_potential": metadata.get('true_potential', 'Unknown'),
                "rich_description": results['documents'][0][i]
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
