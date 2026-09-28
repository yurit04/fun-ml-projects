"""Score every Massive news article with FinBERT (ProsusAI/finbert).

Run once (≈30-60 min on Apple-silicon GPU); the sentiment notebook
(03_sentiment_factor.ipynb) reads the cached output. One parquet shard per
month is written, so the job is resumable.

    ../../.venv/bin/python scripts/score_news_finbert.py
"""
from pathlib import Path
import sys, time

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qdata import MASSIVE, CACHE

OUT = CACHE / "finbert_news"
OUT.mkdir(parents=True, exist_ok=True)
DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
BATCH, MAXLEN = 256, 96

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(DEV).eval()
labels = [model.config.id2label[i] for i in range(3)]          # positive, negative, neutral

for f in sorted((MASSIVE / "curated" / "news").glob("year=*/month=*/data.parquet")):
    y, m = f.parent.parent.name[5:], f.parent.name[6:]
    dst = OUT / f"{y}-{m}.parquet"
    if dst.exists():
        continue
    t0 = time.time()
    d = pd.read_parquet(f, columns=["id", "title", "description"]).drop_duplicates("id")
    text = (d["title"].fillna("") + ". " + d["description"].fillna("").str.slice(0, 400)).tolist()
    probs = []
    with torch.no_grad():
        for i in range(0, len(text), BATCH):
            b = tok(text[i:i + BATCH], padding=True, truncation=True, max_length=MAXLEN,
                    return_tensors="pt").to(DEV)
            probs.append(model(**b).logits.float().softmax(-1).cpu())
    p = torch.cat(probs).numpy()
    res = pd.DataFrame(p, columns=[f"p_{l}" for l in labels])
    res.insert(0, "id", d["id"].values)
    res.to_parquet(dst, index=False)
    print(f"{y}-{m}: {len(d):>6} articles in {time.time() - t0:5.1f}s", flush=True)
