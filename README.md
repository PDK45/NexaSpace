# 🚀 NexaSpace

**The World's First Autonomous Spatial Reasoning Engine for Real Estate**

![NexaSpace Demo Header](https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80)

## 📌 The Problem
Property listing platforms have reached a labeling breaking point. The massive influx of unorganized, multi-perspective room images has outpaced human and basic algorithmic categorization. Traditional static filters (e.g., "3 beds, 2 baths") fail to distinguish subtle architectural nuances, regional design styles, or "cluttered" real-world conditions. This leads to massive search friction and lost revenue.

## 💡 The NexaSpace Solution
NexaSpace is an autonomous layer capable of independent spatial reasoning. It sits between raw image ingestion and the end-user, instantly identifying, high-grading, and strategically mapping images to the correct user intent without human oversight.

Instead of filtering by checkboxes, users search by **semantic emotional intent**:
> _"A kitchen big enough to host Thanksgiving with an industrial vibe, even if it's currently messy."_

Our AI "x-rays" the room, ignores the temporary clutter, and maps the architectural "bones" to the user's exact desire.

---

## 🏗️ Technical Architecture

NexaSpace utilizes a cutting-edge Vision-to-Vector pipeline:

1. **Ingestion (The X-Ray):** We pass raw, unorganized, and highly cluttered real estate photos into **Gemini 1.5 Vision**.
2. **Spatial Reasoning:** We strictly prompt the Vision LLM to act as an Architectural Analyst. It ignores trash and clutter, extracting a structured JSON object containing: `Architectural Style`, `Spatial Flow`, `Lighting Quality`, and `True Potential`.
3. **Vectorization:** The AI generates a "Rich Description" which is embedded and stored in **ChromaDB** (Vector Database) alongside the metadata.
4. **Semantic Search:** A **FastAPI** backend exposes an endpoint that takes a user's natural language query, embeds it, and performs a cosine-similarity search against the ChromaDB, returning flawless matches.

---

## 🛠️ Built With
* **Frontend:** HTML5, Tailwind CSS, Framer Motion (Glassmorphism Premium UI)
* **Backend:** Python, FastAPI, Uvicorn
* **AI & Reasoning:** Google Gemini 1.5 Pro / Flash API
* **Vector Database:** ChromaDB
* **Data Pipelines:** Pandas, Pillow

---

## 🚀 How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.9+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Variables
Rename `.env.example` to `.env` and add your Gemini API Key.
```env
GEMINI_API_KEY="your_api_key_here"
```

### 4. Ingest the Data
To build the vector database, you need to run the ingestion script over your `datasets/` folder.
```bash
python ingest.py
```

### 5. Start the Application
Run the FastAPI server which also serves the gorgeous frontend UI.
```bash
uvicorn app:app --reload
```
Navigate to `http://localhost:8000` in your browser.

---

## 📊 The Dataset Strategy
To prove the enterprise viability of NexaSpace, we utilized a three-tier dataset approach:
1. **The Scale Dataset:** Thousands of standard MLS images to prove volume handling.
2. **The Style Dataset:** Pinterest-curated interior design images to train the model on nuanced architecture (Mid-Century vs. Victorian).
3. **The Chaos Dataset:** The famous "Messy vs. Clean Room" dataset. This proves our ultimate thesis: *Our spatial reasoner can look past severe disorganization and still accurately classify the underlying architectural value.*

---
*Built with ❤️ in 24 Hours.*
