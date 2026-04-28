from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_text_splitters import CharacterTextSplitter


print("=" * 50)
print("LLM Document QA Chatbot")
print("=" * 50)

# Load environment variables if needed later
load_dotenv()

# Connect to the local Ollama model
llm = ChatOllama(model="llama3")

# Load the document that we want to ask questions about
loader = TextLoader("data/sample.txt")
documents = loader.load()

# Split the document into smaller text chunks
text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Convert chunks into vector embeddings and store them in FAISS
embeddings = FakeEmbeddings(size=384)
vector_store = FAISS.from_documents(chunks, embeddings)

# Ask the user for a question
question = input("Ask a question about your document: ")

# Find the most relevant chunks from the document
results = vector_store.similarity_search(question, k=3)
context = "\n".join([doc.page_content for doc in results])

# Give the retrieved context to the LLM and ask it to answer
prompt = f"""
You are a helpful AI assistant.

Answer the user's question using only the context below.
If the answer is not in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

# Show the final answer
print("\n" + "=" * 50)
print("Question:")
print(question)

print("\nAnswer:")
print(response.content)
print("=" * 50)