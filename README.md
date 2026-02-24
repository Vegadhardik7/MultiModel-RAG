# MultiModel-RAG 

![](output-image.png)

# MultiModel-RAG Architecture Flow
                        +---------------------+
                        |   User Interaction   |
                        | (Frontend UI in HTML)|
                        +----------+----------+
                                   |
                                   v
                        +---------------------+
                        |   FastAPI Backend   |
                        |   (app/api routes)  |
                        +----------+----------+
                                   |
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
         +------------------+            +----------------------+
         |   Text Ingestion |            |   Image Ingestion     |
         | (parse uploaded  |            | (parse user images)   |
         |    text files)   |            |                      |
         +--------+---------+            +----------+-----------+
                  |                                   |
                  v                                   v
       +----------------------+           +-----------------------+
       |  Text Embeddings     |           |   Image Embeddings    |
       | (sentence/embed text)|           |   (CLIP or visual)    |
       +-----------+----------+           +-----------+-----------+
                   |                                  |
                   v                                  v
            +-----------------------------------------------+
            |                Vector Store (ChromaDB)         |
            |   (store & index text + image vectors)         |
            +------------------+----------------------------+
                               |
                               v
                  +-------------------------------+
                  |         Retriever Module      |
                  |  (search by query similarity) |
                  +---------------+---------------+
                                  |
                                  v
                     +---------------------------+
                     |      LLM Client Module    |
                     |  (Llama 3.1 8B via Ollama)|
                     +-------------+-------------+
                                   |
                                   v
                      +-------------------------+
                      |   Generated Answer      |
                      |  (RAG Response to User) |
                      +-------------------------+
                                   |
                                   v
                          +-----------------+
                          | Frontend Output |
                          +-----------------+


## 📌 **High-Level Summary of the Project (from Code & Structure)**

**Project Title:** *Llama 3.1-8B-4.7GB + CLIP + chromadb* ([GitHub][1])

This implies the system uses:

* **Local LLM (likely Llama 3.1 8B)** — for generation
* **CLIP model** — for vision/text multimodal embeddings
* **ChromaDB** — as the vector database for retrieval

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
