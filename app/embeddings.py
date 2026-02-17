from sentence_transformers import SentenceTransformer
import numpy as np

# Load small fast local model
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text):
    return np.array(model.encode(text))

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
