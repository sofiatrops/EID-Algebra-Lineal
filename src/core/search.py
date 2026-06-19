from src.core.vectorizer import build_vocabulary, build_tf_matrix
from src.core.similarity import cosine_similarity

def search(corpus_tokens, query_tokens, method="tf" , top_k=5) -> list[tuple[int, float]]:
    vocabulary=build_vocabulary(corpus_tokens)
    matriz=build_tf_matrix(corpus_tokens,vocabulary=vocabulary)
    query_vector=build_tf_matrix([query_tokens],vocabulary)[0]
    scores=[]
    for i in range(len(matriz)):
        sim=cosine_similarity(query_vector,matriz[i])
        scores.append((i,sim))
    scores_descendente=sorted(scores,key=lambda x: x[1], reverse=True)[:top_k] #Ordenamos desde el mayor score hasta el menor (key para agarrar el score)
    return scores_descendente
