from flask import Flask, render_template, jsonify, request
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import system_prompt
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()

# Check for API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# ✅ Optimized lightweight embedding model
def get_embeddings_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

# Initialize components
embeddings = get_embeddings_model()
index_name = "medicalbot"

# Initialize Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up retriever
retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# Initialize LLM using OpenRouter
llm = ChatOpenAI(
    model="openai/gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=1024,
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# Prompt and chains
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def landing():
    return render_template('landing.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def get_chat_response():
    msg = request.form["msg"]
    enhanced_query = f"""
    Please provide a comprehensive medical explanation about {msg}, including:
    1. Definition and key characteristics
    2. Main causes (if applicable)
    3. Common symptoms (if applicable)
    4. Basic treatment approaches (if relevant)
    5. Important medical considerations

    Please structure the response clearly and explain any medical terms used.
    Base your answer strictly on the provided medical documents.
    """
    response = rag_chain.invoke({"input": enhanced_query})
    formatted_answer = response["answer"].replace("\n", "<br>").replace("•", "<br>•")
    return formatted_answer

# Use dynamic port binding for Render
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # <- Use PORT from environment
    app.run(host="0.0.0.0", port=port)

