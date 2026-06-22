import argparse
from pathlib import Path

from core.preprocessing import preprocess
from core.search import search


def load_corpus(corpus_dir: str) -> tuple[list[str], list[str]]:
    paths = sorted(Path(corpus_dir).glob("*.txt"))
    names = []
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        names.append(path.name)
        documents.append(text)
    return names, documents


def print_table(results, names):
    print(f"{'Documento':30s}  {'Score':>6s}")
    print("-" * 40)
    for idx, score in results:
        print(f"{names[idx]:30s}  {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Buscador Semantico Simple")
    parser.add_argument("--corpus", default="data/corpus/",
                        help="Directorio del corpus")
    parser.add_argument("--query", type=str,
                        help="Consulta en lenguaje natural")
    parser.add_argument("--method", choices=["tf", "tfidf"], default="tf",
                        help="Metodo de vectorizacion")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Numero de resultados")
    parser.add_argument("--interactive", action="store_true",
                        help="Modo interactivo")

    args = parser.parse_args()

    names, documents = load_corpus(args.corpus)
    corpus_tokens = [preprocess(doc) for doc in documents]

    def run_query(query_text: str):
        query_tokens = preprocess(query_text)
        results = search(corpus_tokens, query_tokens,
                         method=args.method, top_k=args.top_k)
        print(f"\nResultados ({args.method.upper()}):")
        print_table(results, names)

    if args.interactive:
        print("Modo interactivo. Escribe 'salir' para terminar.")
        while True:
            try:
                q = input("\nConsulta: ").strip()
            except EOFError:
                break
            if not q or q.lower() == "salir":
                break
            run_query(q)
    elif args.query:
        run_query(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
