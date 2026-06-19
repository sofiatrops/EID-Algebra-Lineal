import numpy as np

def crear_vocabulario(documents):
    palabras_unicas = set()
    for documento in documents:
        for palabra in documento:
            palabras_unicas.add(palabra)
    vocabulario = list(palabras_unicas)
    vocabulario.sort()
    return vocabulario

def crear_matriz_tf(documents, vocabulary):
    num_documentos = len(documents)
    num_palabras = len(vocabulary)
    matriz_tf = np.zeros((num_documentos, num_palabras))
    
    palabra_a_indice = {}
    for i in range(num_palabras):
        palabra = vocabulary[i]
        palabra_a_indice[palabra] = i
        
    for i in range(num_documentos):
        documento = documents[i]
        total_palabras = len(documento)
        if total_palabras == 0:
            continue
        for palabra in documento:
            if palabra in palabra_a_indice:
                indice = palabra_a_indice[palabra]
                matriz_tf[i, indice] = matriz_tf[i, indice] + 1.0
                
        for j in range(num_palabras):
            conteo_bruto = matriz_tf[i, j]
            matriz_tf[i, j] = conteo_bruto / total_palabras
            
    return matriz_tf

def crear_matriz_tfidf(documents, vocabulary, usar_suavizado=True) -> np.ndarray:
    matriz_tf = crear_matriz_tf(documents, vocabulary)
    num_documentos = len(documents)
    num_palabras = len(vocabulary)
    
    frecuencia_documento = np.zeros(num_palabras)
    for j in range(num_palabras):
        conteo = 0.0
        for i in range(num_documentos):
            if matriz_tf[i, j] > 0:
                conteo = conteo + 1.0
        frecuencia_documento[j] = conteo

    vector_idf = np.zeros(num_palabras)
    for j in range(num_palabras):
        if usar_suavizado:
            denominador = frecuencia_documento[j] + 1.0
            vector_idf[j] = np.log(num_documentos / denominador)
        else:
            if frecuencia_documento[j] > 0:
                vector_idf[j] = np.log(num_documentos / frecuencia_documento[j])
            else:
                vector_idf[j] = 0.0

    matriz_tfidf = matriz_tf * vector_idf
    
    return matriz_tfidf
