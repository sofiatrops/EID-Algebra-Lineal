import numpy as np


def build_vocabulary(documents: list[list[str]]) -> list[str]:
    vocabulary = []
    documents = sum(documents, [])
    for token in documents:
        if token not in vocabulary:
            vocabulary.append(token)
    return sorted(vocabulary)


def build_tf_matrix(documents: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    matrix = np.zeros((len(documents), len(vocabulary)))
    for i in range(len(documents)):
        for j in range(len(vocabulary)):
            matrix[i, j] = documents[i].count(vocabulary[j])
    return matrix


# TF-IDF = TF normalizado * IDF. El IDF baja el peso de las palabras que
# aparecen en muchos documentos (poco distintivas) y sube el de las raras.
def build_tfidf_matrix(documents: list[list[str]], vocabulary: list[str], smoothing: bool = True) -> np.ndarray:
    n_docs = len(documents)
    n_terms = len(vocabulary)

    # TF normalizado: frecuencia relativa de cada termino dentro de su documento
    tf = np.zeros((n_docs, n_terms))
    for i in range(n_docs):
        total = len(documents[i])
        if total == 0:
            continue
        for j in range(n_terms):
            tf[i, j] = documents[i].count(vocabulary[j]) / total

    # IDF: log(N / df). df = en cuantos documentos aparece el termino
    idf = np.zeros(n_terms)
    for j in range(n_terms):
        df = np.count_nonzero(tf[:, j])
        if smoothing:
            idf[j] = np.log(n_docs / (df + 1.0))  # +1 evita dividir por cero
        elif df > 0:
            idf[j] = np.log(n_docs / df)

    return tf * idf