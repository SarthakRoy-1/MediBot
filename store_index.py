from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

def main():
    # Load environment variables from .env file or Render dashboard
    load_dotenv()

    # Get Pinecone API key and environment from env
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    
    if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
        raise ValueError("Missing Pinecone API key or environment")

    # Step 1: Load and process PDF data
    print("Loading PDF files...")
    extracted_data = load_pdf_file(data='Data/')  # Adjust path if needed

    print("Splitting text into chunks...")
    text_chunks = text_split(extracted_data)

    print("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()

    # Step 2: Initialize Pinecone
    print("Initializing Pinecone client...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medicalbot"  # Same as in app.py

    # Step 3: Create index if it doesn't exist
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating Pinecone index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine"
        )

    # Step 4: Get the index
    index = pc.Index(index_name)

    # Step 5: Store the text chunks in Pinecone
    print("Storing documents in Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print("✅ Index creation and storage completed successfully!")

if __name__ == "__main__":
    main()
