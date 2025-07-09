import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from langchain_qdrant import QdrantVectorStore
from langchain.chains import RetrievalQA,retrieval_qa
from langchain_groq import ChatGroq


QDRANT_COLLECTION = #your collectio name
API_KEY = #your qdrant api key
QDRANT_URL = #your qdrant url
GROQ_API_KEY = # Put your Groq API key here


qdrant_client = QdrantClient(url=QDRANT_URL, api_key=API_KEY)

def scrape_webpage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return " ".join(soup.stripped_strings)

urls = [
    "https://qdrant.tech/documentation/",
    "https://python.langchain.com/docs/get_started/introduction",
    "https://openai.com/research"
]

all_text = ""
for url in urls:
    print(f"Scraping: {url}")
    content = scrape_webpage(url)
    all_text += f"\n\nFrom {url}:\n{content}"


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_text(all_text)

embedder = SpacyEmbeddings(model_name="en_core_web_md")

embeddings = []
documents = []
for i, chunk in enumerate(chunks):
    embedding = embedder.embed_documents([chunk])[0]
    embeddings.append(embedding)
    documents.append(Document(page_content=chunk))
    # print(f"Chunk {i + 1} embedded. First 5 dims: {embedding[:5]}")

# --- CREATE COLLECTION IF NOT EXISTS ---
if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
    )

# --- UPLOAD POINTS ---
qdrant_client.upsert(
    collection_name=QDRANT_COLLECTION,
    points=[
        PointStruct(id=i, vector=embeddings[i], payload={"text": documents[i].page_content})
        for i in range(len(embeddings))
    ]
)


vectorstore = QdrantVectorStore(
    client=qdrant_client,
    collection_name=QDRANT_COLLECTION,
    embedding=embedder
)

retriever = vectorstore.as_retriever()

llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)


qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


query = "What is an LLM?"
answer = qa_chain.run(query)

print("Query:", query)
print("Answer:", answer)
