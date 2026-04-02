"""
Load song data into MongoDB and build the FAISS index.

Usage: python build_index.py
"""

import logging
import os
import time

import faiss
import numpy as np
from pymongo import MongoClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

META_FILE = "data/meta_word2vec_2M.tsv"
EMB_FILE = "data/emb_word2vec_2M.tsv"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "spotify_recommendations"
COLLECTION = "songs"
FAISS_INDEX_FILE = "faiss_index.bin"
BATCH_SIZE = 10_000
EMBEDDING_DIM = 32


def parse_meta_line(line: str) -> tuple[str, str]:
    """Parse 'Song Title- Artist Name' format."""
    text = line.strip()
    parts = text.rsplit("- ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return text, "Unknown Artist"


def flush_batch(
    collection,
    index: faiss.Index,
    batch_docs: list[dict],
    batch_embeddings: list[list[float]],
) -> None:
    collection.insert_many(batch_docs, ordered=False)
    emb_array = np.array(batch_embeddings, dtype=np.float32)
    faiss.normalize_L2(emb_array)
    index.add(emb_array)


def main() -> None:
    logger.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]
    collection.drop()
    logger.info("Database: %s, Collection: %s", DB_NAME, COLLECTION)

    logger.info("Counting rows...")
    with open(META_FILE, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f)
    logger.info("Total songs: %s", f"{total_rows:,}")

    logger.info("Creating FAISS index (dim=%d)...", EMBEDDING_DIM)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

    logger.info("Loading data into MongoDB + FAISS index...")
    start_time = time.time()
    batch_docs: list[dict] = []
    batch_embeddings: list[list[float]] = []
    total_loaded = 0

    with open(META_FILE, "r", encoding="utf-8") as meta_f, \
         open(EMB_FILE, "r", encoding="utf-8") as emb_f:
        for idx, (meta_line, emb_line) in enumerate(
            tqdm(zip(meta_f, emb_f), total=total_rows, desc="Loading")
        ):
            title, artist = parse_meta_line(meta_line)
            emb_values = [float(x) for x in emb_line.strip().split("\t")]
            batch_docs.append({"_id": idx, "title": title, "artist": artist})
            batch_embeddings.append(emb_values)

            if len(batch_docs) >= BATCH_SIZE:
                flush_batch(collection, index, batch_docs, batch_embeddings)
                total_loaded += len(batch_docs)
                batch_docs = []
                batch_embeddings = []

    if batch_docs:
        flush_batch(collection, index, batch_docs, batch_embeddings)
        total_loaded += len(batch_docs)

    elapsed = time.time() - start_time
    logger.info("Loaded %s songs in %.1fs", f"{total_loaded:,}", elapsed)

    logger.info("Creating text indexes...")
    collection.create_index([("title", "text"), ("artist", "text")])
    collection.create_index("artist")
    logger.info("Text index on title + artist created.")

    logger.info("Saving FAISS index to %s...", FAISS_INDEX_FILE)
    faiss.write_index(index, FAISS_INDEX_FILE)
    fsize = os.path.getsize(FAISS_INDEX_FILE) / (1024 * 1024)
    logger.info("FAISS index size: %.1f MB (%s vectors)", fsize, f"{index.ntotal:,}")

    logger.info("Verification:")
    logger.info("  MongoDB docs: %s", f"{collection.count_documents({}):,}")
    logger.info("  FAISS vectors: %s", f"{index.ntotal:,}")
    logger.info("  Sample doc: %s", collection.find_one({"_id": 0}))


if __name__ == "__main__":
    main()
