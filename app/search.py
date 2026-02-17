from .embeddings import get_embedding, cosine_similarity
from .data import documents

# Precompute embeddings at startup
for doc in documents:
    doc["embedding"] = get_embedding(doc["content"])

def vector_search(query, k=10):
    query_embedding = get_embedding(query)

    scores = []

    for doc in documents:
        score = cosine_similarity(query_embedding, doc["embedding"])

        # normalize cosine (-1 to 1) → (0 to 1)
        normalized = (score + 1) / 2

        scores.append((doc, normalized))

    scores.sort(key=lambda x: x[1], reverse=True)

    return scores[:k]
