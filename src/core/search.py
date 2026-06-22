from core.vectorizer import build_vocabulary, build_tf_matrix,build_idf_matrix
from core.similarity import cosine_similarity

def search(corpus_tokens, query_tokens, method="tf" |"idf" , top_k=5) -> list[tuple[int, float]]:
    vocabulary=build_vocabulary(corpus_tokens)
    matriz_tf=build_tf_matrix(corpus_tokens,vocabulary=vocabulary)
    query_vector_tf=build_tf_matrix([query_tokens],vocabulary)[0]
    if method=="tf":
        matriz=build_tf_matrix(corpus_tokens,vocabulary=vocabulary)
        query_vector=build_tf_matrix([query_tokens],vocabulary)[0]
    elif method=="idf":
        idf_vector = build_idf_matrix(corpus_tokens, vocabulary=vocabulary)
        # Aplicamos el IDF como un multiplicador (NumPy lo aplica fila por fila automáticamente)
        matriz = matriz_tf * idf_vector
        query_vector = query_vector_tf * idf_vector
    else:
        raise ValueError("El método debe ser 'tf' o 'idf'")
    scores=[]
    for i in range(len(matriz)):
        sim=cosine_similarity(query_vector,matriz[i])
        scores.append((i,sim))
    scores_descendente=sorted(scores,key=lambda x: x[1], reverse=True)[:top_k] #Ordenamos desde el mayor score hasta el menor (key para agarrar el score)
    return scores_descendente
