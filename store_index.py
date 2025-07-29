from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    
    # Load and process PDF data
    print("Loading PDF files...")
    extracted_data = load_pdf_file(data='Data/')
    
    print("Splitting text into chunks...")
    text_chunks = text_split(extracted_data)
    
    print("Loading embeddings...")
    embeddings = download_hugging_face_embeddings()
    
    # Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    index_name = "medicalbot"
    
    # Create index if it doesn't exist
    existing_indexes = [index.name for index in pc.list_indexes()]
    if index_name not in existing_indexes:
        print("Creating new Pinecone index...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine"
        )
    
    # Get the index
    index = pc.Index(index_name)
    
    # Store documents in Pinecone
    print("Storing documents in Pinecone...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        index_name=index_name
    )
    
    print("Index creation completed successfully!")

if __name__ == "__main__":
    main()
