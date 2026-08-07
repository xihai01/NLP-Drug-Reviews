"""
Interactive query interface for the drug-review BM25 search engine.

Usage
-----
    python search_drugs.py [path_to_csv] [--top_k N]

Then type queries at the prompt (e.g. "anxiety insomnia sleep problems"),
one per line. Type 'quit' or 'exit' (or Ctrl-D) to stop.
"""
import sys
import argparse
import textwrap
from Lab5_IR_bm25_skeleton import BM25Index

DEFAULT_CSV = "./data/druglib_outputs/processed/druglib_processed_combined.csv"


def build_index(csv_path):
    print(f"Loading reviews from {csv_path} ...")
    docs = BM25Index.load_druglib_csv(csv_path)
    print(f"Loaded {len(docs)} reviews. Building BM25 index ...")
    index = BM25Index(docs)
    docs_by_id = {d["id"]: d for d in docs}
    print(f"Ready. {index.info()}\n")
    return index, docs_by_id


def print_results(results, docs_by_id, top_k):
    if not results:
        print("  No matching reviews found.\n")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        doc = docs_by_id[doc_id]
        meta = doc["meta"]
        snippet = textwrap.shorten(doc["text"], width=160, placeholder=" ...")
        print(f"{rank}. [{score:.3f}] {meta['drug']}  (condition: {meta['condition']})")
        print(f"   rating={meta['rating']}  effectiveness={meta['effectiveness']}")
        print(f"   \"{snippet}\"\n")


def run_repl(index, docs_by_id, top_k):
    print("Enter a search query (or 'quit' to exit).")
    while True:
        try:
            query = input("\nsearch> ").strip()
        except EOFError:
            print()
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        results = index.score(query, top_k=top_k)
        print_results(results, docs_by_id, top_k)


def run_demo(index, docs_by_id, top_k):
    """Non-interactive fallback: runs a few sample queries so the script is
    useful even when there's no attached terminal (e.g. piped input)."""
    demo_queries = [
        "anxiety insomnia sleep problems",
        "nausea vomiting stomach pain",
        "birth control weight gain",
    ]
    for q in demo_queries:
        print(f"search> {q}")
        results = index.score(q, top_k=top_k)
        print_results(results, docs_by_id, top_k)


def main():
    parser = argparse.ArgumentParser(description="Search drug reviews with BM25.")
    parser.add_argument("csv_path", nargs="?", default=DEFAULT_CSV,
                         help=f"Path to druglib_processed_*.csv (default: {DEFAULT_CSV})")
    parser.add_argument("--top_k", type=int, default=5, help="Number of results to show (default: 5)")
    args = parser.parse_args()

    index, docs_by_id = build_index(args.csv_path)

    if sys.stdin.isatty():
        run_repl(index, docs_by_id, args.top_k)
    else:
        run_demo(index, docs_by_id, args.top_k)


if __name__ == "__main__":
    main()
