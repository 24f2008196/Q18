from sentence_transformers import CrossEncoder

# Better model for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, candidates, rerank_k=6):
    pairs = [(query, doc["content"]) for doc, _ in candidates]

    scores = cross_encoder.predict(pairs)

    reranked = []

    for i, (doc, _) in enumerate(candidates):
        # normalize scores roughly between 0-1
        score = float(scores[i])
        normalized = (score - min(scores)) / (max(scores) - min(scores) + 1e-6)


        reranked.append((doc, normalized))

    reranked.sort(key=lambda x: x[1], reverse=True)

    return reranked[:rerank_k]
