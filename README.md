# Buscador Semántico Simple

**Proyecto de Investigación 2 — Álgebra Lineal para la Computación**
EID 2026 · Entrega: 23 de junio de 2026, 23:59

---

## ¿Qué vamos a construir?

Un buscador semántico simple que, dada una consulta en lenguaje natural, devuelve los documentos de un corpus que son más parecidos a la consulta — usando **álgebra lineal**.

La idea: representar cada texto como un **vector** en un espacio de muchas dimensiones (una por cada palabra del vocabulario), y comparar la consulta contra cada documento midiendo el **coseno del ángulo** entre sus vectores. Cuanto más cerca de 1 está el coseno, más parecidos son los textos.

### Lo que el programa hace, paso a paso

1. **Carga el corpus** (16 textos en español, organizados en 4 temas).
2. **Preprocesa** los textos: pasa todo a minúsculas, quita tildes, divide en palabras, elimina stopwords ("el", "la", "que", ...).
3. **Construye un vocabulario** con todas las palabras únicas.
4. **Vectoriza** cada documento con dos métodos:
   - **TF** (frecuencia de términos)
   - **TF-IDF** (TF ponderado por la rareza de cada palabra)
5. **Recibe una consulta** del usuario, la vectoriza igual.
6. **Calcula la similitud coseno** entre la consulta y cada documento.
7. **Devuelve los top-K documentos más relevantes** ordenados por score.

### Lo que el informe analiza

- Por qué representar textos como vectores tiene sentido.
- Por qué el coseno funciona para medir similitud semántica.
- Cómo afecta el tamaño del vocabulario al rendimiento.
- Ventajas y limitaciones del enfoque (no entiende sinónimos, sensible al vocabulario, etc.).
- Visualizaciones: mapa de calor de similitudes, frecuencia de términos, comparación TF vs TF-IDF.

---

## Estructura del repositorio

```
EID-Algebra-Lineal/
├── README.md                  ← este archivo
├── requirements.txt           ← dependencias Python
├── .gitignore
│
├── src/                       ← código fuente
│   ├── preprocessing.py       ← normalizar, tokenizar, quitar stopwords
│   ├── vectorizer.py          ← build_vocabulary, build_tf_matrix, build_tfidf_matrix
│   ├── similarity.py          ← producto punto, norma, similitud coseno
│   ├── search.py              ← orquesta corpus → vector → ranking
│   ├── heatmap.py             ← visualización de matriz de similitudes
│   ├── term_frequency.py      ← visualización de frecuencias
│   └── main.py                ← CLI con argparse
│
├── tests/                     ← pruebas unitarias con pytest
│   ├── test_similarity.py     ← casos numéricos exactos (cos=1, cos=0, cos=0.5)
│   ├── test_vectorizer.py
│   └── test_search.py
│
├── data/
│   └── corpus/                ← 16 textos .txt en español (4 por tema)
│
├── notebooks/
│   └── analisis_resultados.ipynb  ← experimentos y gráficos para el informe
│
└── docs/
    ├── informe.tex            ← informe LaTeX
    ├── informe.pdf            ← versión final
    ├── presentacion.pptx      ← slides de la defensa
    └── figuras/               ← imágenes generadas para el informe
```

---

## Equipo y roles

| Integrante | GitHub | Rol primario |
|---|---|---|
| **Denys Rodríguez** *(líder)* | [@sofiatrops](https://github.com/sofiatrops) | Project Manager · Integrador · Compilación del informe |
| **Paulo Villalobos** | [@notKechai](https://github.com/notKechai) | Matemático · Marco teórico · Redacción del informe |
| **Joaquín Valenzuela** | [@Joaco20x](https://github.com/Joaco20x) | Core developer · Implementación matemática · Tests |
| **David** | [@David7985](https://github.com/David7985) | Corpus · CLI · Visualizaciones · Notebook de análisis |

> Las preguntas de la defensa son aleatorias a cualquier integrante. Antes del 22 hacemos una sesión de **Knowledge Transfer** para que todos puedan defender cualquier parte.

---

## Cómo correr el proyecto

### Requisitos
- Python 3.10 o superior
- Windows / Mac / Linux

### Instalación

```bash
git clone https://github.com/sofiatrops/EID-Algebra-Lineal.git
cd EID-Algebra-Lineal
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
```

### Uso

**Búsqueda con consulta directa:**
```bash
python -m src.main --corpus data/corpus/ --query "fútbol champions league"
```

**Modo interactivo:**
```bash
python -m src.main --interactive
```

**Cambiar método de vectorización:**
```bash
python -m src.main --query "..." --method tfidf   # default: tf
```

### Tests

```bash
pytest tests/ -v
```

### Notebook de análisis

```bash
jupyter notebook notebooks/analisis_resultados.ipynb
```

---

## Convención de trabajo

### Ramas

- `main` — protegida, solo merges vía PR
- `develop` — integración diaria
- `feat/<nombre>` — una rama por feature (vida máx. 24 h, mergear rápido evita conflictos)

### Commits

Formato [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(vectorizer): implementa matriz TF con vocabulario único (#6)
fix(similarity): corrige división por cero en vectores nulos (#7)
docs(informe): agrega ejemplo numérico de similitud coseno (#9)
test(similarity): añade caso de vectores ortogonales (#7)
```

Incluir `(#N)` con el número del issue al que pertenece el commit. Cuando un commit dice `Closes #N`, el issue se cierra solo al mergear.

### Pull Requests

- Una PR por feature, con descripción clara
- Al menos 1 review antes de mergear a `develop`
- `develop` → `main` solo en momentos de integración (D3, D5)

---

## Sprint de 5 días

| Día | Fecha | Foco |
|---|---|---|
| **D1** | 18 jun | Setup, corpus, primer borrador del marco teórico |
| **D2** | 19 jun | Preprocessing + TF + similitud coseno + Sección I del informe |
| **D3** | 20 jun | TF-IDF + buscador funcional + CLI + Sección II |
| **D4** | 21 jun | Análisis, visualizaciones, Sección III, **Knowledge Transfer** |
| **D5** | 22 jun | Secciones IV+V, presentación, **ensayo cronometrado + Q&A cruzado** |
| **D6** | 23 jun | Entrega antes de 23:59 |

El tablero completo con los 23 issues está en **Projects** del repo.

---

## Stack técnico

| Tecnología | Para qué |
|---|---|
| Python 3.10+ | Lenguaje principal |
| NumPy | Núcleo matemático (vectores, matrices, operaciones) |
| Matplotlib + Seaborn | Visualizaciones |
| pytest | Tests unitarios |
| Jupyter | Notebook de análisis |
| scikit-learn | **Solo** para validación cruzada en el notebook (no en el código principal) |
| LaTeX | Informe final |

> **Decisión clave:** la matemática se implementa **a mano con NumPy**, sin scikit-learn. Es un proyecto de Álgebra Lineal — el evaluador quiere ver que dominamos las operaciones vectoriales, no que sabemos llamar a una librería.

---

## Reglas del equipo

- Todos los integrantes tienen commits propios y significativos (en código y/o docs).
- Nada de copiar literal desde una IA — el PDF del curso lo penaliza explícitamente.
- Si alguien se traba, lo dice **el mismo día** en el grupo, no a último momento.
- Todo lo que se discute en reuniones se anota en el issue correspondiente.

---

## Referencias

- Enunciado oficial: `docs/EID_Lineal.pdf`
- Curso: Álgebra Lineal para la Computación
- Entrega: plataforma del curso, antes del 23/06/2026 23:59
