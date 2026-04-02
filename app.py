"""
Song Recommendation Web App (Flask Backend).

Usage: python app.py
Then open http://localhost:5000 in your browser.
"""

import logging
import os

import faiss
import numpy as np
from flask import Flask, jsonify, render_template, request
from pymongo import MongoClient
from pymongo.collection import Collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "spotify_recommendations")
COLLECTION = os.environ.get("COLLECTION", "songs")
FAISS_INDEX_FILE = os.environ.get("FAISS_INDEX_FILE", "faiss_index.bin")
EMB_FILE = os.environ.get("EMB_FILE", "data/emb_word2vec_2M.tsv")
EMBEDDING_DIM = 32

app = Flask(__name__)


def load_resources() -> tuple[Collection, faiss.Index, np.ndarray]:
    logger.info("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION]
    logger.info("%s songs in database", f"{collection.count_documents({}):,}")

    logger.info("Loading FAISS index...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    logger.info("%s vectors loaded", f"{index.ntotal:,}")

    logger.info("Loading embeddings into memory...")
    embeddings = np.zeros((index.ntotal, EMBEDDING_DIM), dtype=np.float32)
    with open(EMB_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            embeddings[i] = [float(x) for x in line.strip().split("\t")]
    faiss.normalize_L2(embeddings)
    logger.info("Embeddings loaded: %s", embeddings.shape)

    return collection, index, embeddings


collection, index, embeddings = load_resources()
logger.info("Ready! Starting web server...")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/search")
def search_songs():
    """Search songs by title or artist."""
    q = request.args.get("q", "").strip()
    if not q or len(q) < 2:
        return jsonify([])

    results = collection.find(
        {"$text": {"$search": q}},
        {"score": {"$meta": "textScore"}},
    ).sort([("score", {"$meta": "textScore"})]).limit(20)

    songs = [
        {"id": doc["_id"], "title": doc["title"], "artist": doc["artist"]}
        for doc in results
    ]

    if not songs:
        regex_results = collection.find({
            "$or": [
                {"title": {"$regex": q, "$options": "i"}},
                {"artist": {"$regex": q, "$options": "i"}},
            ]
        }).limit(20)
        songs = [
            {"id": doc["_id"], "title": doc["title"], "artist": doc["artist"]}
            for doc in regex_results
        ]

    return jsonify(songs)


@app.route("/api/recommend")
def recommend():
    """Get similar songs using FAISS vector similarity."""
    song_id = request.args.get("id", type=int)
    n = min(request.args.get("n", 20, type=int), 50)

    if song_id is None or song_id < 0 or song_id >= index.ntotal:
        return jsonify({"error": "Invalid song ID"}), 400

    query_vec = embeddings[song_id].reshape(1, -1).copy()
    distances, neighbor_ids = index.search(query_vec, n + 1)

    recommendations = []
    for dist, neighbor_id in zip(distances[0], neighbor_ids[0]):
        neighbor_id = int(neighbor_id)
        if neighbor_id == song_id:
            continue
        doc = collection.find_one({"_id": neighbor_id})
        if doc:
            recommendations.append({
                "id": neighbor_id,
                "title": doc["title"],
                "artist": doc["artist"],
                "similarity": round(float(dist), 4),
            })

    return jsonify(recommendations[:n])


@app.route("/api/random")
def random_songs():
    """Get random songs for discovery."""
    songs = [
        {"id": doc["_id"], "title": doc["title"], "artist": doc["artist"]}
        for doc in collection.aggregate([{"$sample": {"size": 12}}])
    ]
    return jsonify(songs)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug, port=5000)
