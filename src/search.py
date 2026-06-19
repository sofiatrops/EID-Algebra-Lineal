from vectorizer import build_vocabulary, build_tf_matrix
from core.similarity import cosine_similarity  

def search(corpus_tokens, query_tokens, method="tf" , top_k=5) -> list[tuple[int, float]]:
    vocabulary=build_vocabulary(corpus_tokens)
    matriz=build_tf_matrix(corpus_tokens,vocabulary=vocabulary)
    query_vector=build_tf_matrix([query_tokens],vocabulary)[0]
    scores=[]
    for i in range(len(matriz)):
        sim=cosine_similarity(query_vector,matriz[i])
        scores.append((i,sim))
    scores_descendente=sorted(scores,key=lambda x: x[1], reverse=True)[:top_k]
    return scores_descendente
    
if __name__ == "__main__":
    corpus = [
        ["algebra", "lineal", "vectores", "matrices"],
        ["buscador", "texto", "palabras", "busqueda"],
        ["algebra", "vectores", "busqueda", "computacion"],
        ["redes", "computacion", "sistemas", "datos"],
        ["texto", "palabras", "documentos", "busqueda"],
    ]

    query = ["algebra", "vectores"]

    resultados = search(corpus, query, method="tf", top_k=3)
    print("Resultados:")
    print(resultados)