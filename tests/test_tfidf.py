import numpy as np
import pytest
from src.vectorizer import crear_matriz_tfidf

def test_tfidf_calculo_a_mano():
    documentos = [
        ["gato", "come"],
        ["perro", "come"]
    ]
    vocabulario = ["come", "gato", "perro"]
    
    tfidf_sin_suave = crear_matriz_tfidf(documentos, vocabulario, usar_suavizado=False)
    
    esperado_sin = np.zeros((2, 3))
    esperado_sin[0, 0] = 0.5 * np.log(2.0 / 2.0)
    esperado_sin[0, 1] = 0.5 * np.log(2.0 / 1.0)
    esperado_sin[0, 2] = 0.0 * np.log(2.0 / 1.0)
    esperado_sin[1, 0] = 0.5 * np.log(2.0 / 2.0)
    esperado_sin[1, 1] = 0.0 * np.log(2.0 / 1.0)
    esperado_sin[1, 2] = 0.5 * np.log(2.0 / 1.0)
    
    np.testing.assert_allclose(tfidf_sin_suave, esperado_sin, rtol=1e-7)
    
    tfidf_con_suave = crear_matriz_tfidf(documentos, vocabulario, usar_suavizado=True)
    
    esperado_con = np.zeros((2, 3))
    esperado_con[0, 0] = 0.5 * np.log(2.0 / (2.0 + 1.0))
    esperado_con[0, 1] = 0.5 * np.log(2.0 / (1.0 + 1.0))
    esperado_con[0, 2] = 0.0 * np.log(2.0 / (1.0 + 1.0))
    esperado_con[1, 0] = 0.5 * np.log(2.0 / (2.0 + 1.0))
    esperado_con[1, 1] = 0.0 * np.log(2.0 / (1.0 + 1.0))
    esperado_con[1, 2] = 0.5 * np.log(2.0 / (1.0 + 1.0))
    
    np.testing.assert_allclose(tfidf_con_suave, esperado_con, rtol=1e-7)

def test_tfidf_documentos_vacios():
    documentos = [
        [],
        ["gato"]
    ]
    vocabulario = ["gato"]
    
    matriz = crear_matriz_tfidf(documentos, vocabulario, usar_suavizado=True)
    
    esperado = np.zeros((2, 1))
    esperado[0, 0] = 0.0
    esperado[1, 0] = 1.0 * np.log(2.0 / (1.0 + 1.0))
    
    np.testing.assert_allclose(matriz, esperado, rtol=1e-7)

def test_tfidf_comparacion_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return
        
    documentos_texto = [
        "gato come",
        "perro come"
    ]
    documentos = []
    for texto in documentos_texto:
        palabras = texto.split()
        documentos.append(palabras)
        
    vocabulario = ["come", "gato", "perro"]
    
    vectorizador = TfidfVectorizer(vocabulary=vocabulario, norm=None, smooth_idf=False)
    tfidf_sklearn = vectorizador.fit_transform(documentos_texto).toarray()
    
    nuestro_tfidf = crear_matriz_tfidf(documentos, vocabulario, usar_suavizado=False)
    
    num_documentos = len(documentos)
    num_palabras = len(vocabulario)
    for i in range(num_documentos):
        palabras = documentos[i]
        total_palabras = len(palabras)
        for j in range(num_palabras):
            palabra_actual = vocabulario[j]
            conteo_bruto = 0.0
            for palabra in palabras:
                if palabra == palabra_actual:
                    conteo_bruto = conteo_bruto + 1.0
            
            valor_sklearn = tfidf_sklearn[i, j]
            valor_esperado = 0.0
            if total_palabras > 0:
                valor_esperado = (valor_sklearn - conteo_bruto) / total_palabras
                
            valor_nuestro = nuestro_tfidf[i, j]
            assert abs(valor_nuestro - valor_esperado) < 1e-7
