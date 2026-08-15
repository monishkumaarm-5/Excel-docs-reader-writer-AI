#download_embedding_model.py
"""
One-time setup script: downloads and caches the local embedding model.

Run this ONCE, with internet access:

    python -m doxl_ai_terminal.data_handler.download_embedding_model

It pulls "sentence-transformers/all-MiniLM-L6-v2" into the project's
own cache folder (src-relative: vector_store/../models/embeddings).
After this finishes, doxl-ai-terminal's embeddings run with ZERO
network access — get_embedding_model() forces offline mode and only
ever reads from this local cache.

You only need to re-run this if you delete the cache folder or want
to switch to a different embedding model.
"""

import os

# Make sure THIS run is allowed to hit the network — it's the one
# script whose whole job is to populate the offline cache.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)


def main():
    from doxl_ai_terminal.data_handler.vector_config import (
        EMBEDDING_MODEL_NAME,
        EMBEDDING_CACHE_DIR,
    )
    from sentence_transformers import SentenceTransformer

    print(f"Downloading '{EMBEDDING_MODEL_NAME}' ...")
    print(f"Caching to: {EMBEDDING_CACHE_DIR}")

    SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=EMBEDDING_CACHE_DIR)

    print("\nDone. The embedding model is now cached locally.")
    print("From now on, get_embedding_model() runs fully offline —")
    print("no network access needed for embeddings.")


if __name__ == "__main__":
    main()
