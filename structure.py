import os

# Define the base project directory
base_dir = "./"

# Define the folder structure
folders = [
    os.path.join(base_dir, "data", "docs"),
    os.path.join(base_dir, "data", "images"),
    os.path.join(base_dir, "chroma_db"),
    os.path.join(base_dir, "app", "ingest"),
    os.path.join(base_dir, "app", "embeddings"),
    os.path.join(base_dir, "app", "vectorstore"),
    os.path.join(base_dir, "app", "llm"),
    os.path.join(base_dir, "scripts"),
]

# Define the files to create
files = [
    os.path.join(base_dir, "README.md"),
    os.path.join(base_dir, "requirements.txt"),
    os.path.join(base_dir, ".env"),
    os.path.join(base_dir, "app", "ingest", "text_ingest.py"),
    os.path.join(base_dir, "app", "ingest", "image_ingest.py"),
    os.path.join(base_dir, "app", "embeddings", "text_embedder.py"),
    os.path.join(base_dir, "app", "embeddings", "image_embedder.py"),
    os.path.join(base_dir, "app", "vectorstore", "chroma_client.py"),
    os.path.join(base_dir, "app", "retriever.py"),
    os.path.join(base_dir, "app", "llm", "ollama_client.py"),
    os.path.join(base_dir, "app", "rag_pipeline.py"),
    os.path.join(base_dir, "app", "api.py"),
    os.path.join(base_dir, "scripts", "setup_check.py"),
]

# Create directories
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file in files:
    with open(file, "w", encoding="utf-8") as f:
        f.write("")  # leave empty or add boilerplate if needed

print("Project structure created successfully!")
