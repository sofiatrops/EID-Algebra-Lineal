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


# IDF: log(N / df). df = en cuantos documentos aparece cada termino.
def build_idf_matrix(documents: list[list[str]], vocabulary: list[str], smoothing: bool = True) -> np.ndarray:
    idf = np.zeros(len(vocabulary))
    tf = build_tf_matrix(documents, vocabulary)
    for j in range(len(vocabulary)):
        df = np.count_nonzero(tf[:, j])
        if smoothing:
            idf[j] = np.log(len(documents) / (df + 1.0))  # +1 evita dividir por cero
        elif df > 0:
            idf[j] = np.log(len(documents) / df)
    return idf


# TF-IDF = TF normalizado * IDF. El IDF baja el peso de las palabras que
# aparecen en muchos documentos (poco distintivas) y sube el de las raras.
def build_tfidf_matrix(documents: list[list[str]], vocabulary: list[str], smoothing: bool = True) -> np.ndarray:
    tf = build_tf_matrix(documents, vocabulary)
    # TF normalizado: frecuencia relativa de cada termino dentro de su documento
    totales = tf.sum(axis=1, keepdims=True)
    totales[totales == 0] = 1.0  # documentos vacios: evita dividir por cero
    tf_norm = tf / totales
    idf = build_idf_matrix(documents, vocabulary, smoothing)
    return tf_norm * idf
