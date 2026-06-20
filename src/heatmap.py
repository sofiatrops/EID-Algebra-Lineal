import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_similarity_heatmap(sim_matrix: np.ndarray,
                            doc_names: list[str],
                            save_path: str = None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix,
                xticklabels=doc_names,
                yticklabels=doc_names,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                square=True)
    plt.title("Matriz de Similitudes entre Documentos")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return plt.gcf()
