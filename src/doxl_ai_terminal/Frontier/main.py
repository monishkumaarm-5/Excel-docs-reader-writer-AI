from doxl_ai_terminal.pipeline.pipeliner import process_file
from doxl_ai_terminal.pipeline.search import search


def main():
    filepath = input("Paste file path: ").strip().strip('"')

    print(f"\nProcessing: {filepath}")

    instances = process_file(filepath)

    print("\n" + "=" * 50)
    print("VECTOR DB INSTANCES")
    print("=" * 50)

    for db in instances:
        print(
            f"  [{db.label}] {db.format_name} "
            f"→ {db.collection_name} ({db.total_chunks} chunks)"
        )

    query = input("\nSearch query (or Enter to skip): ").strip()

    if query:
        search(instances, query)


if __name__ == "__main__":
    main()