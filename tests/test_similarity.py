import math

import numpy as np
import pytest

from src.core.similarity import (
    cosine_similarities,
    cosine_similarity,
    dot_product,
    norm,
)


# ----------------------------- producto punto -----------------------------

def test_dot_product_valor_conocido():
    # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32.0


def test_dot_product_vectores_ortogonales_es_cero():
    assert dot_product([1, 0], [0, 1]) == 0.0


def test_dot_product_dimensiones_distintas_lanza_error():
    with pytest.raises(ValueError):
        dot_product([1, 2, 3], [1, 2])


# --------------------------------- norma ----------------------------------

def test_norm_triangulo_3_4_5():
    # ||(3, 4)|| = sqrt(9 + 16) = 5
    assert norm([3, 4]) == 5.0


def test_norm_vector_nulo_es_cero():
    assert norm([0, 0, 0]) == 0.0


# ----------------------------- similitud coseno ---------------------------

def test_coseno_igual_a_1_vectores_paralelos():
    # Mismo sentido (uno es multiplo del otro) -> angulo 0 -> cos = 1
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_coseno_igual_a_0_vectores_ortogonales():
    # Angulo de 90 grados -> cos = 0
    assert cosine_similarity([1, 0], [0, 5]) == pytest.approx(0.0)


def test_coseno_igual_a_0_5_angulo_60_grados():
    # (1, 0) y (1, sqrt(3)): dot = 1, normas 1 y 2 -> cos = 1/2
    assert cosine_similarity([1, 0], [1, math.sqrt(3)]) == pytest.approx(0.5)


def test_coseno_simetrico():
    u, v = [1, 2, 3], [3, 2, 1]
    assert cosine_similarity(u, v) == pytest.approx(cosine_similarity(v, u))


def test_coseno_acotado_entre_0_y_1_con_frecuencias():
    # Vectores de frecuencias (>= 0): el coseno nunca sale de [0, 1]
    s = cosine_similarity([0, 1, 1, 0, 1], [1, 1, 0, 1, 0])
    assert 0.0 <= s <= 1.0
    # Comparten solo "come" -> 1/9 dividido por (1/sqrt(3))^2 = 1/3
    assert s == pytest.approx(1 / 3)


def test_coseno_con_vector_nulo_devuelve_0():
    # Sin la guarda esto seria una division por cero
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0


def test_coseno_ambos_nulos_devuelve_0():
    assert cosine_similarity([0, 0], [0, 0]) == 0.0


# -------------------------- similitud contra matriz ------------------------

def test_cosine_similarities_contra_matriz():
    query = [1, 0, 0]
    matriz = [
        [1, 0, 0],  # identico -> 1
        [0, 1, 0],  # ortogonal -> 0
        [2, 0, 0],  # paralelo -> 1
    ]
    resultado = cosine_similarities(query, matriz)
    assert resultado == pytest.approx(np.array([1.0, 0.0, 1.0]))
