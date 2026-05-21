import chromadb
from sentence_transformers import SentenceTransformer

print("creating or retrieving a chroma db")

client = chromadb.PersistentClient(
    path="./chroma_db"
)

print("creating a collection")

collection = client.get_or_create_collection(
    name="games"
)

print("loading sentence transformer")

model = SentenceTransformer(
    "all-MiniLM-L6-v2",
)

documents = [
    {
        "id": "1",
        "text": "Cyberpunk futuristic racing game"
    },
    {
        "id": "2",
        "text": "Fantasy RPG with dragons"
    }
]

print("inserting docs")

for doc in documents:

    embedding = model.encode(
        doc["text"]
    ).tolist()

    collection.add(
        ids=[doc["id"]],
        embeddings=[embedding],
        documents=[doc["text"]]
    )


query = "dark futuristic action game"

print("converting sentence into vector")

query_embedding = model.encode(
    query
).tolist()

print("querying chromadb")

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nResults:\n")

for doc in results["documents"][0]:
    print(doc)