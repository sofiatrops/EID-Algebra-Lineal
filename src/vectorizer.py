import numpy as np

"""
doc1 = ["algebra", "lineal", "algebra"]
doc2 = ["buscador", "lineal", "texto"]

          algebra  buscador  lineal  texto
doc1  →  [  2,       0,       1,      0  ]
doc2  →  [  0,       1,       1,      1  ]

"""

def build_vocabulary(documents: list[list[str]]) -> list[str]:
    vocabulary=[]
    documents= sum(documents,[])
    for frase in documents:
        if frase not in vocabulary:
            vocabulary.append(frase)
    return sorted(vocabulary)
def build_tf_matrix(documents: list[list[str]], vocabulary: list[str]) -> np.ndarray:
    matriz=np.zeros((len(documents),len(vocabulary)))
    for i in range(len(documents)):
        for j in range(len(vocabulary)):
            matriz[i,j]=documents[i].count(vocabulary[j])
    return matriz
doc1 = ["algebra", "lineal", "algebra"]
doc2 = ["buscador", "lineal", "texto"]

vocab = build_vocabulary([doc1, doc2])
matriz = build_tf_matrix([doc1, doc2], vocab)

print(vocab)
print(matriz)