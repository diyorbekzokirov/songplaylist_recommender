"""
Inspect the raw TSV data files (meta + embeddings).

Usage: python inspect_data.py
"""

import logging
import os

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FILES = {
    "meta": "data/meta_word2vec_2M.tsv",
    "emb": "data/emb_word2vec_2M.tsv",
}
PREVIEW_ROWS = 5


def inspect_file(label: str, filename: str) -> None:
    logger.info("FILE [%s]: %s", label, filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    logger.info("Size: %.1f MB", size_mb)

    with open(filename, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= PREVIEW_ROWS:
                break
            cols = line.strip().split("\t")
            logger.info("Row %d: (%d columns)", i, len(cols))
            for j, col in enumerate(cols):
                preview = col[:80] + ("..." if len(col) > 80 else "")
                logger.info("  col[%d]: %s", j, preview)

    logger.info("Counting rows...")
    with open(filename, "r", encoding="utf-8") as f:
        total = sum(1 for _ in f)
    logger.info("Total rows: %s\n", f"{total:,}")


def main() -> None:
    for label, filename in FILES.items():
        inspect_file(label, filename)


if __name__ == "__main__":
    main()
