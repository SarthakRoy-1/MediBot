import os
from pinecone import Pinecone
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
# OR Option 2 (recommended): use environment variable
# os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"

# Create Pinecone client
pc = Pinecone(api_key=api_key)

# Get your index (check it exists first)
index_name = "us-east-1"  # replace this with your index name

if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' not found.")
else:
    index = pc.Index(index_name)

# Now you're ready to use the index (e.g., query, upsert)
print("Connected to index:", index_name)
