# NexaSpace Architecture

This diagram visualizes the autonomous spatial reasoning pipeline. You can use this directly in your hackathon presentation.

```mermaid
graph TD
    %% Styling
    classDef userLayer fill:#f9f9f9,stroke:#333,stroke-width:2px,color:#000;
    classDef processingLayer fill:#e1bee7,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef dbLayer fill:#b2ebf2,stroke:#0097a7,stroke-width:2px,color:#000;
    classDef aiLayer fill:#c8e6c9,stroke:#388e3c,stroke-width:2px,color:#000;

    subgraph User Experience [Frontend - React/Tailwind]
        A[User Input: Natural Language Intent]:::userLayer
        B[NexaSpace Web App]:::userLayer
        C[Visual Results & Spatial Insights]:::userLayer
    end

    subgraph API Gateway [FastAPI Backend]
        D[Intent Semantic Searching]:::processingLayer
        E[Image Ingestion Service]:::processingLayer
    end

    subgraph AI Engine [Vision & Reasoning]
        F[Raw Messy Image Data]:::aiLayer
        G[Gemini 1.5 Pro Vision Model\n'Architectural Analyst']:::aiLayer
        H[JSON Spatial Embedding\nStyle, Flow, Potential]:::aiLayer
    end

    subgraph Data Store [Vector Database]
        I[(ChromaDB / Pinecone)]:::dbLayer
        J[Vector Embeddings]:::dbLayer
    end

    %% Ingestion Flow
    F -->|Uploads| E
    E -->|API Call + Prompt| G
    G -->|Extracts Structure| H
    H -->|Stores Metadata & Text| I
    I -->|Generates Vectors| J

    %% Search Flow
    A -->|Types 'Industrial but messy'| B
    B -->|POST /api/search| D
    D -->|Query Text| I
    I -->|Cosine Similarity Match| D
    D -->|Returns Top 5 Matches| C
```

## How to use this:
You can copy the code block above into a Mermaid Live Editor (https://mermaid.live/) or directly into Notion/GitHub to instantly generate a professional architecture slide for your pitch deck.
