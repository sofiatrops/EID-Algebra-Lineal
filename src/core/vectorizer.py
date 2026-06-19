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