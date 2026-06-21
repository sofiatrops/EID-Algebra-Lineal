import numpy as np


# Producto punto: u . v = sum(u_i * v_i)
def dot_product(u, v) -> float:
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    if u.shape != v.shape:
        raise ValueError(f"Los vectores deben tener la misma dimension: {u.shape} vs {v.shape}")
    return float(np.sum(u * v))


# Norma euclidiana (largo del vector): ||v|| = sqrt(sum(v_i^2))
def norm(v) -> float:
    v = np.asarray(v, dtype=float)
    return float(np.sqrt(np.sum(v ** 2)))


# Similitud coseno: cos(theta) = (u . v) / (||u|| ||v||). Mide el angulo, no el largo.
# Si un vector es nulo la division no esta definida -> devolvemos 0.0.
def cosine_similarity(u, v) -> float:
    norm_u = norm(u)
    norm_v = norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return dot_product(u, v) / (norm_u * norm_v)


# Similitud coseno de una consulta contra cada fila (documento) de una matriz.
def cosine_similarities(query, matrix) -> np.ndarray:
    query = np.asarray(query, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    return np.array([cosine_similarity(query, fila) for fila in matrix])
