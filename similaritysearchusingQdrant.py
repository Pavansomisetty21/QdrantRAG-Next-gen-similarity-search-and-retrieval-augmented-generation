from qdrant_client.models import VectorParams, Distance, PointStruct
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
QDRANT_COLLECTION = "collection name"
api_key=""#qdrant api key
url=""#your qdrant url
# --- SETUP QDRANT ---
qdrant_client = QdrantClient(
    url=url,
    api_key=api_key
)

# --- SCRAPE WEBPAGES ---
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

if not qdrant_client.collection_exists(collection_name=QDRANT_COLLECTION):
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=len(embeddings[0]),
            distance=Distance.COSINE
        )
    )

# 2. Upload points to Qdrant
qdrant_client.upsert(
    collection_name=QDRANT_COLLECTION,
    points=[
        PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"text": documents[i].page_content}
        )
        for i in range(len(embeddings))
    ]
)
query = "What is an llm?"

# Get the embedding for your query
query_vector = embedder.embed_query(query)  # SpacyEmbeddings uses `embed_query`

# Search
results = qdrant_client.search(
    collection_name=QDRANT_COLLECTION,
    query_vector=query_vector,
    limit=5
)

for res in results:
    print(f"Score: {res.score:.3f}")
    print(f"Text: {res.payload['text'][:200]}")
    print("-" * 40)

