from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
#from langchain_google_genai import ChatGoogleGenerativeAI
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

# Check for Google API key
#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#if not GOOGLE_API_KEY:
#    raise ValueError("GOOGLE_API_KEY not found in environment variables")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables")

# Initialize components
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

# Initialize Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Set up retriever with supported parameters
retriever = docsearch.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 8}  # Only use 'k' parameter
)

# Initialize newer Gemini model with better parameters
# llm = ChatGoogleGenerativeAI(
#     #model="gemini-2.0-pro-exp",  # Using the latest experimental model
#     model="models/gemini-2.0-flash-exp",
#     temperature=0.3,  # Reduced for more focused responses
#     max_tokens=1024,  # Increased token limit for more detailed responses
#     google_api_key=GOOGLE_API_KEY,
#     convert_system_message_to_human=True,
#     top_p=0.8,  # Added for better response quality
#     top_k=40    # Added for better response diversity
# )

llm = ChatOpenAI(
    #model="google/gemini-2.5-pro",
    model="openai/gpt-3.5-turbo", # <--- CHOOSE YOUR OPENROUTER MODEL HERE (see Step 5)
    temperature=0.3,
    max_tokens=1024,
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1" # This is crucial for OpenRouter
)

# Create prompt template and chains
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def landing():
    return render_template('landing.html')

@app.route("/chat")
def chat():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def get_chat_response():  # Renamed from 'chat' to 'get_chat_response'
    msg = request.form["msg"]
    
    # Enhance the user query with specific instructions
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
    
    # Get response using enhanced query
    response = rag_chain.invoke({"input": enhanced_query})
    
    # Format the response for better readability
    formatted_answer = response["answer"].replace("\n", "<br>")
    formatted_answer = formatted_answer.replace("•", "<br>•")
    
    return formatted_answer



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)  #--> if in development phase
