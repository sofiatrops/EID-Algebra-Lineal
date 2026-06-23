import matplotlib.pyplot as plt
import numpy as np


def plot_top_terms(matrix: np.ndarray,
                   vocabulary: list[str],
                   top_n: int = 20,
                   save_path: str = None):
    doc_freqs = matrix.sum(axis=0)
    indices = np.argsort(doc_freqs)[::-1][:top_n]

    words = [vocabulary[i] for i in indices]
    freqs = doc_freqs[indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(words)), freqs, color="steelblue")
    plt.yticks(range(len(words)), words)
    plt.xlabel("Frecuencia")
    plt.title(f"Top-{top_n} Terminos mas Frecuentes")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()
