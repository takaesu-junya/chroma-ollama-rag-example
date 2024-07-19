import json
import chromadb
from chromadb.utils import embedding_functions


ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.Client()
collection = client.get_or_create_collection(name="collection_name", embedding_function=ef)


collection.add(
    documents=["document1", "document2", "document3"],
    metadatas=[{"type": "recipe"}, {"type": "article"}, {"type": "article"}],
    ids=["1", "2", "3"]
)

results = collection.query(
    query_texts=["document"],
    n_results=2,
    where={
        "type": {
            "$eq": "recipe"
        }
    }
)

print(json.dumps(results, indent=4))
