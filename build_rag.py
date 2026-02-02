import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from termcolor import colored
from config import get_embeddings, DATA_FOLDER, DB_FOLDER, FILES

def build_vector_dbs():
    """
    ETL Pipeline: Load PDF -> Clean -> Split -> Embed -> Store in ChromaDB
    """
    embeddings = get_embeddings()
    
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
        print(colored(f"❌ Error: {DATA_FOLDER} directory not found.", "red"))
        return

    for key, filename in FILES.items():
        persist_dir = os.path.join(DB_FOLDER, key)
        file_path = os.path.join(DATA_FOLDER, filename)

        # 檢查是否已經存在 DB，如果存在就跳過 (避免重複建立)
        if os.path.exists(persist_dir):
            print(colored(f"✅ DB for '{key}' already exists at {persist_dir}. Skipping...", "yellow"))
            continue

        if not os.path.exists(file_path):
            print(colored(f"❌ Missing source file: {filename}", "red"))
            continue

        print(colored(f"🔨 Building Vector Index for {key}...", "cyan"))
        
        # 1. Load PDF
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()
        print(f"   - Loaded {len(docs)} pages.")

        # ==================================================================
        # [Student Area] Data Cleaning & Splitting Strategy
        # ==================================================================
        
        # TODO: Clean the data (e.g., remove newlines)
        for doc in docs:
            doc.page_content = doc.page_content.replace("\n", " ")

        # TODO: Tune the chunking strategy
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,     # Experiment with this!
            chunk_overlap=500,   # Experiment with this!
            separators=["\n\n", "\n", " ", ""] 
        )
        # ==================================================================

        splits = splitter.split_documents(docs)
        print(f"   - Split into {len(splits)} chunks.")
        
        # 2. Embed & Store
        print("   - Embedding and storing... (This may take a while)")
        Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
        print(colored(f"🎉 Successfully built DB for {key}!", "green"))

if __name__ == "__main__":
    build_vector_dbs()