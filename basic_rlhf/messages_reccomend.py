from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding


qdrant_client = QdrantClient(
    url="http://localhost:6333"
    )

collection_name = "messages"

model_dense = TextEmbedding(
    cache_dir="fastembed"
)

def upsert_message(action, message, feedback, recreate=True):
    dense_embeddings = list(model_dense.embed(message))
    if recreate: 
        qdrant_client.recreate_collection(
            collection_name,
            vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
        )

    count = qdrant_client.count(collection_name).count
    
    qdrant_client.upsert(
        collection_name,
        points=[models.PointStruct(
            id=count,
            payload={
                "action": action,
                "response": message,
                "feedback": feedback
            },
            vector=dense_embeddings[0].tolist()
        )],
    )
    print("UPser done!")

def retrieve_message(message):
    # Perform the search
    points = qdrant_client.query_points(
        collection_name,
        query=list(model_dense.embed(message))[0].tolist(),
        limit=20,
        with_vectors=False,
        # score_threshold=.5
    ).points

    positives = []
    negatives = []
    for point in points:
        if point.payload["feedback"] != 1:
            negatives.append(point)
        else:
            positives.append(point)
    
    return "\n".join(set([
        f"Respuestas positivas:\n{r.payload["action"]}: {r.payload["response"]}" 
        if point.payload["feedback"] == 1 else f"Respuestas negativas:\n{r.payload["action"]}: {r.payload["response"]}" 
        for r in points]))


if __name__ == "__main__":
    # upsert_message("Precio del brunch", "320$ por persona", 1, True)
    pass