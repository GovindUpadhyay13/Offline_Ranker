#!/usr/bin/env python3
"""Offline precompute (network allowed): vendor the models and build caches.

Embeds every candidate's evidence, builds the FAISS and BM25 indexes, and
persists vectors, both indexes, and the candidate_id order so rank.py can run
with no network. Run once. Optional argv[1] overrides the data path for a smoke
test on a smaller file.
"""

import json
import os
import sys
import time

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from src import features, io, retrieval


def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else config.DATA_PATH
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    t = time.perf_counter()
    ids, texts = [], []
    for c in io.read_candidates(data_path):
        ids.append(c.candidate_id)
        texts.append(features.evidence_text(c))
    print(f"load: {len(ids)} candidates, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    embedder = SentenceTransformer(config.EMBED_MODEL)
    embedder.save(config.EMBED_DIR)
    CrossEncoder(config.CROSS_ENCODER_MODEL).save(config.CROSS_DIR)
    AutoTokenizer.from_pretrained(config.FLAN_MODEL).save_pretrained(config.FLAN_DIR)
    AutoModelForSeq2SeqLM.from_pretrained(config.FLAN_MODEL).save_pretrained(config.FLAN_DIR)
    print(f"models: vendored to {config.MODELS_DIR}, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    emb = embedder.encode(
        texts,
        batch_size=config.EMBED_BATCH,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(config.EMB_NPY, emb)
    print(f"embed: {emb.shape}, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    retrieval.build_dense_index(emb, config.FAISS_PATH)
    print(f"faiss: index built, {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    retrieval.build_bm25(texts, config.BM25_PATH)
    print(f"bm25: index built, {time.perf_counter() - t:.1f}s")

    with open(config.IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"ids: {len(ids)} written to {config.IDS_PATH}")

    # Build lookup table of founding years for companies appearing in the candidate pool
    founding_years = {
        'Accenture': 1989, 'Acme Corp': 1900, 'Adobe': 1982, 'Aganitha': 2017,
        'Amazon': 1994, 'Apple': 1976, "BYJU'S": 2011, 'CRED': 2018,
        'Capgemini': 1967, 'Cognizant': 1994, 'Dream11': 2008, 'Dunder Mifflin': 1949,
        'Flipkart': 2007, 'Freshworks': 2010, 'Genpact AI': 1997, 'Glance': 2019,
        'Globex Inc': 1900, 'Google': 1998, 'HCL': 1976, 'Haptik': 2013,
        'Hooli': 1900, 'InMobi': 2007, 'Infosys': 1981, 'Initech': 1900,
        'Krutrim': 2023, 'LinkedIn': 2002, 'Locobuzz': 2015, 'Mad Street Den': 2013,
        'Meesho': 2015, 'Meta': 2004, 'Microsoft': 1975, 'Mindtree': 1999,
        'Mphasis': 1998, 'Netflix': 1997, 'Niramai': 2016, 'Nykaa': 2012,
        'Observe.AI': 2017, 'Ola': 2010, 'Paytm': 2010, 'PharmEasy': 2015,
        'PhonePe': 2015, 'Pied Piper': 2010, 'PolicyBazaar': 2008, 'Razorpay': 2014,
        'Rephrase.ai': 2019, 'Saarthi.ai': 2017, 'Salesforce': 1999, 'Sarvam AI': 2023,
        'Stark Industries': 1900, 'Swiggy': 2014, 'TCS': 1968, 'Tech Mahindra': 1986,
        'Uber': 2009, 'Unacademy': 2015, 'Vedantu': 2011, 'Verloop.io': 2015,
        'Wayne Enterprises': 1900, 'Wipro': 1945, 'Wysa': 2015, 'Yellow.ai': 2016,
        'Zoho': 1996, 'Zomato': 2008, 'upGrad': 2015
    }
    with open(config.FOUNDING_YEARS_PATH, "w", encoding="utf-8") as f:
        json.dump(founding_years, f, indent=2)
    print(f"founding_years: written to {config.FOUNDING_YEARS_PATH}")


if __name__ == "__main__":
    main()
