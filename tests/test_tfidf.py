"""Pruebas unitarias para build_tfidf_matrix en src/core/vectorizer.py.

Verifican el calculo de TF-IDF (TF normalizado * IDF) con valores exactos
calculados a mano, el caso de documentos vacios y una comparacion contra
scikit-learn (solo si esta instalado).
"""

import numpy as np

from src.core.vectorizer import build_tfidf_matrix


def test_tfidf_calculo_a_mano():
    documentos = [
        ["gato", "come"],
        ["perro", "come"],
    ]
    vocabulario = ["come", "gato", "perro"]

    # --- Sin suavizado: idf = log(N / df) ---
    tfidf_sin_suave = build_tfidf_matrix(documentos, vocabulario, smoothing=False)
    esperado_sin = np.zeros((2, 3))
    esperado_sin[0, 0] = 0.5 * np.log(2.0 / 2.0)  # "come" esta en los 2 docs -> idf 0
    esperado_sin[0, 1] = 0.5 * np.log(2.0 / 1.0)  # "gato" solo en doc 0
    esperado_sin[1, 0] = 0.5 * np.log(2.0 / 2.0)
    esperado_sin[1, 2] = 0.5 * np.log(2.0 / 1.0)  # "perro" solo en doc 1
    np.testing.assert_allclose(tfidf_sin_suave, esperado_sin, rtol=1e-7)

    # --- Con suavizado: idf = log(N / (df + 1)) ---
    tfidf_con_suave = build_tfidf_matrix(documentos, vocabulario, smoothing=True)
    esperado_con = np.zeros((2, 3))
    esperado_con[0, 0] = 0.5 * np.log(2.0 / (2.0 + 1.0))
    esperado_con[0, 1] = 0.5 * np.log(2.0 / (1.0 + 1.0))
    esperado_con[1, 0] = 0.5 * np.log(2.0 / (2.0 + 1.0))
    esperado_con[1, 2] = 0.5 * np.log(2.0 / (1.0 + 1.0))
    np.testing.assert_allclose(tfidf_con_suave, esperado_con, rtol=1e-7)


def test_tfidf_documentos_vacios():
    documentos = [
        [],
        ["gato"],
    ]
    vocabulario = ["gato"]

    matriz = build_tfidf_matrix(documentos, vocabulario, smoothing=True)
    esperado = np.zeros((2, 1))
    esperado[0, 0] = 0.0  # documento vacio -> fila en cero, sin division por cero
    esperado[1, 0] = 1.0 * np.log(2.0 / (1.0 + 1.0))
    np.testing.assert_allclose(matriz, esperado, rtol=1e-7)


def test_tfidf_comparacion_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return  # sklearn no esta instalado: se omite la comparacion

    documentos_texto = ["gato come", "perro come"]
    documentos = [t.split() for t in documentos_texto]
    vocabulario = ["come", "gato", "perro"]

    vectorizador = TfidfVectorizer(vocabulary=vocabulario, norm=None, smooth_idf=False)
    tfidf_sklearn = vectorizador.fit_transform(documentos_texto).toarray()
    nuestro_tfidf = build_tfidf_matrix(documentos, vocabulario, smoothing=False)

    # sklearn usa idf = ln(N/df) + 1 y conteo bruto; normalizamos para comparar
    for i in range(len(documentos)):
        total = len(documentos[i])
        for j in range(len(vocabulario)):
            conteo = documentos[i].count(vocabulario[j])
            valor_sklearn = tfidf_sklearn[i, j]
            esperado = (valor_sklearn - conteo) / total if total > 0 else 0.0
            assert abs(nuestro_tfidf[i, j] - esperado) < 1e-7
