from src.core.vectorizer import build_vocabulary, build_tf_matrix, build_idf_matrix
from src.core.similarity import cosine_similarity


def search(corpus_tokens, query_tokens, method="tf", top_k=5) -> list[tuple[int, float]]:
    vocabulary = build_vocabulary(corpus_tokens)
    matriz = build_tf_matrix(corpus_tokens, vocabulary)
    query_vector = build_tf_matrix([query_tokens], vocabulary)[0]
    if method == "idf":
        # Ponderamos cada termino por su IDF (NumPy lo aplica columna por columna)
        idf_vector = build_idf_matrix(corpus_tokens, vocabulary)
        matriz = matriz * idf_vector
        query_vector = query_vector * idf_vector
    elif method != "tf":
        raise ValueError("El metodo debe ser 'tf' o 'idf'")
    scores = []
    for i in range(len(matriz)):
        sim = cosine_similarity(query_vector, matriz[i])
        scores.append((i, sim))
    # Ordenamos de mayor a menor score y devolvemos los top_k
    return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
