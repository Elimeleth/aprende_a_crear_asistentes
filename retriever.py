from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

import os
from dotenv import load_dotenv
load_dotenv()

# Load example document
with open("data.txt") as f:
    data = f.read()

def loader():
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
    )
    docs = text_splitter.create_documents([data])
    return [d.page_content for d in docs]


qdrant_client = QdrantClient(
    url="https://2c8786bd-b5e1-42b7-9fb9-594cf4b0489e.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY")
    )

collection_name = "gogh"

model_dense = TextEmbedding(
    cache_dir="fastembed"
)

model_sparse = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1",
    cache_dir="fastembed"
    )

def upload():
    texts = loader()
    dense_embeddings = list(model_dense.embed(texts, batch_size=10))
    sparse_embeddings = (list(map(lambda x: dict(indices=x.indices, values=x.values), list(model_sparse.embed(texts, batch_size=10)))))

    qdrant_client.recreate_collection(
        collection_name,
        vectors_config={
            "dense-vector": models.VectorParams(
                size=384,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            )
        },
         sparse_vectors_config={
            "text": models.SparseVectorParams(),
        },
    )


    BATCH_SIZE = 10  # Adjust this based on your data and Qdrant limits

    for i in range(0, len(texts), BATCH_SIZE):
        batch_ids = [id for id in range(i, min(i + BATCH_SIZE, len(texts)))]
        batch_payloads = [{"text": text} for text in texts[i: i + BATCH_SIZE]]
        batch_dense_embeddings = dense_embeddings[i: i + BATCH_SIZE]
        batch_sparse_embeddings = sparse_embeddings[i: i + BATCH_SIZE]

        qdrant_client.upsert(
            collection_name,
            points=models.Batch(
                ids=batch_ids,
                payloads=batch_payloads,
                vectors={
                    "dense-vector": batch_dense_embeddings,
                    "text": batch_sparse_embeddings,
                },
            ),
        )
        print(f"Uploaded batch {i // BATCH_SIZE + 1}") # Optional: Print progress

def qdrant_retrieval(query: str) -> str:
    """
    Busca informacion relacionada a la query del cliente.

    Args:
      query: La query que el cliente ha realizado.

    Returns:
      Un string con la informacion relacionada a la query.
    """
    
    sparse_text = list(model_sparse.query_embed(query))
    try:
        points = qdrant_client.query_points(
          collection_name,
          prefetch=[
              models.Prefetch(
                  query=models.SparseVector(indices=sparse_text[0].indices,
                        values=sparse_text[0].values
                    ),
                  using="text",
                  limit=50,
                  score_threshold=.35,
              )
          ],
          query=list(model_dense.embed([query])),
          using="dense-vector",
          limit=5,
          score_threshold=.7,
        ).points
        
        return [Document(page_content=point.payload["text"]) for point in points]
        # return "\n".join(list([point.payload["text"] for point in points]))
    except Exception as e:
        return str(e)
  
if __name__ == "__main__":
    pass