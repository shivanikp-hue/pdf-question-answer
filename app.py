from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

print("STARTING APP...")

# Load PDF
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# Split text into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Create embeddings (FREE)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in FAISS vector DB
db = FAISS.from_documents(texts, embeddings)

print("PDF processed successfully!")

# Better FREE model (instruction-based)
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

# Ask user question
query = input("Ask a question from the PDF: ")

# Retrieve relevant chunks
docs = db.similarity_search(query)
context = " ".join([doc.page_content for doc in docs])

# Create prompt (important for good answers)
prompt = f"""
You are a helpful assistant. Answer the question clearly based only on the context.

Context:
{context}

Question:
{query}

Answer:
"""

# Generate answer
result = qa_pipeline(prompt, max_length=200)

print("\nAnswer:")
print(result[0]["generated_text"])