from typing import List

from doxl_ai_terminal.data_structure.excel import VectorDBInstance


def search(db_instances: List[VectorDBInstance], query: str):
    """Search across all vector stores"""

    for db in db_instances:
        print(f"\n  [{db.label}] {db.format_name}:")

        # Method 1: Using retriever
        results = db.retriever.invoke(query)

        for doc in results:
            print(f"    → {doc.page_content[:80]}")
            print(f"      {doc.metadata}")

        # Method 2: Using similarity_search_with_score
        scored = db.vectorstore.similarity_search_with_score(query, k=3)

        print(f"\n  With scores:")
        for doc, score in scored:
            print(f"    → [{score:.4f}] {doc.page_content[:80]}")