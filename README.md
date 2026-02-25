# MultiModel-RAG 

![](output-image.png)

# MultiModel-RAG Architecture Flow

                        ┌──────────────────────────┐
                        │   User uploads PDF       │
                        └──────────────┬───────────┘
                                       │
                                       ▼
                   ┌──────────────────────────────────┐
                   │ partition_pdf() (Unstructured)   │
                   │ - Text (NarrativeText, Title)    │
                   │ - Tables                         │
                   │ - Images                         │
                   └──────────────┬───────────────────┘
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
  Text Elements             Table Elements          Image Elements
  (len > 80)                _linearize_table()      _describe_image()
                                                    │
                                                    └─ CLIP (ViT-B/32)
                                                       → Visual label
                                                       → Image context text
          
          └───────────────────────┬────────────────────────┘
                                  ▼
                        Combined Text Chunks
                 (Text / "Table: ..." / "Image context: ...")
                                  │
                                  ▼
                     embed_texts() → SentenceTransformer
                     (all-mpnet-base-v2)
                                  │
                                  ▼
                        Chroma Vector Store
                (Session-specific PersistentClient)
                                  │
                                  ▼
                         ─────────RAG PHASE─────────
                                  │
User Query ───────────────────────┘
      │
      ▼
embed_texts(query)
      │
      ▼
retrieve() from Chroma (top k=6)
      │
      ▼
build_prompt()
  - Inject retrieved chunks
  - Inject conversation history
      │
      ▼
Ollama LLM (generate / generate_stream)
      │
      ▼
Final Answer
      │
      ▼
Stored in Session Memory (last 4 turns)

## 📌 **High-Level Summary of the Project (from Code & Structure)**

**Project Title:** *Llama 3.1-8B-4.7GB + CLIP + chromadb* ([GitHub][1])

This implies the system uses:

* **Local LLM (likely Llama 3.1 8B)** — for generation
* **CLIP model** — for vision/text multimodal embeddings
* **ChromaDB** — as the vector database for retrieval

### **Explain the complete flow of this MultiModel RAG application.**  

This application is built around a multimodal retrieval pipeline with four main stages: **ingestion, embedding, retrieval, and generation**.  

1. **Ingestion:**  
   The system starts by taking in a PDF document. Using the `partition_pdf()` function from the Unstructured library, the document is broken down into three types of content: text, tables, and images.  
   - Text is cleaned and filtered based on length.  
   - Tables are converted into linearized text so they can be processed like normal text.  
   - Images go through a visual understanding pipeline using the CLIP model (ViT-B/32). CLIP generates descriptive text from the image, making it searchable in the same space as text and tables.  

2. **Embedding:**  
   Once all content is converted into text chunks (plain text, table text, and image descriptions), these chunks are embedded using the SentenceTransformer model `all-mpnet-base-v2`. The embeddings are stored in a **Chroma vector database**, with each session having its own persistent client to ensure isolation.  

3. **Retrieval:**  
   When a user asks a question, the query itself is embedded using the same SentenceTransformer model. The system then performs a similarity search in Chroma to find the most relevant chunks (top-k results).  

4. **Generation:**  
   The retrieved chunks are combined with the user’s recent conversation history to form a structured prompt. This prompt is sent to the Ollama LLM, which generates a response. The output can be streamed or returned as a complete answer, and the conversation history is updated for continuity.  

**In summary:**  
The system takes text, tables, and images from a PDF, converts them into a unified embedding space, stores them in a vector database, retrieves the most relevant information when queried, and finally uses an LLM to generate contextual answers. This makes the application capable of handling multimodal inputs while delivering precise, conversational outputs.  

---

## 📁 **Repository Structure**

The main folders visible in the project are:

* **`app/`** — application source code (backend logic) ([GitHub][1])
* **`chroma_sessions/`** — likely stores session or vector store states ([GitHub][1])
* **`frontend/`** — front-end UI (HTML/JS) for interacting with the app ([GitHub][1])
* **`memory/`** — memory storage/caching mechanisms ([GitHub][1])
* **`scripts/`** — utility scripts (install/setup helpers) ([GitHub][1])
* **`output-image.png`** — image preview showing sample output behavior ([GitHub][1])
* **`requirements.txt`** — Python dependencies ([GitHub][2])
* **`structure.py`** — script to generate the project skeleton ([GitHub][3])

---

## 🧠 **Dependencies (From `requirements.txt`)**

The project uses the following libraries (implying system capabilities): ([GitHub][2])

* **FastAPI** + **uvicorn** — backend API server
* **Requests, pillow** — HTTP and image handling
* **sentence-transformers** / **torch / transformers** — text embedding models
* **chromadb** — vector database
* **unstructured** — document parsing (esp. for PDFs/text)
* **python-magic-bin** / **huggingface_hub** — file metadata + HF model downloads

---

## 🧱 **Inferred System Modules (From `structure.py`)**

Although these files aren’t fully visible, the structure builder shows what functional modules *would* exist: ([GitHub][3])

### Ingestion

* `text_ingest.py` → for ingesting text docs
* `image_ingest.py` → for ingesting images

### Embedding

* `text_embedder.py` → text embeddings
* `image_embedder.py` → multimodal (vision) embeddings

### Vector Store

* `chroma_client.py` → connects to ChromaDB

### Retriever

* `retriever.py` → retrieves relevant chunks from ChromaDB

### LLM Interface

* `ollama_client.py` → interface to local LLM (likely Llama)

### RAG Pipeline

* `rag_pipeline.py` → puts together embed → retrieve → generate

### API

* `api.py` → FastAPI routes for frontend/backend interaction

### Auxiliary

* `setup_check.py` → checks environment setup

---

## 📊 **Putting It All Together — *What the System Does***

From the structure and naming alone, we can infer the **data flow**:

1. **Ingestion**

   * User uploads text/image files via the frontend.
   * The system parses ingestion files (`text_ingest`, `image_ingest`).

2. **Embedding**

   * Text sections converted to embeddings using sentence models.
   * Images converted to vector embeddings using CLIP.

3. **Storage**

   * Embeddings saved into **ChromaDB**.

4. **Retrieval**

   * For any user query, the retriever fetches the top relevant vectors from Chroma.

5. **Generation**

   * Retrieved context passed to the LLM (via `ollama_client`) for answer generation.

6. **API / UI**

   * Frontend sends user request to FastAPI backend, which responds with results.

This is a classic **Multimodal Retrieval-Augmented Generation (RAG) pipeline**, combining vision + text modalities. ([Medium][4])
