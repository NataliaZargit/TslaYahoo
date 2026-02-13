# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# import libraries
import json, time, csv, sys, random
from pathlib import Path
from typing import Dict, Any, List
import requests
from datetime import datetime, timezone
from dateutil import parser as dtparser

API_URL = "https://api-2-0.spot.im/v1.0.0/conversation/read"

# TSLA forum constants (Yahoo Finance - OpenWeb)
SPOT_ID = "sp_Rba9aFpG"
POST_ID = "finmb$27444752"
CONVERSATION_ID = f"{SPOT_ID}_{POST_ID}"

PAGE_SIZE = 250
SLEEP_BETWEEN = 0.2
MAX_RETRIES = 6

OUT_CSV  = Path("content/tsla_conversations_filtered")
OUT_JSON = Path("content/tsla_conversations_filtered")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:131.0) "
        "Gecko/20100101 Firefox/131.0"
    ),
    "Content-Type": "application/json",
    "x-spot-id": SPOT_ID,
    "x-post-id": POST_ID,
}
--- Date filter: only messages from 21 Oct 2024 onward ---
FILTER_START = datetime(2024, 10, 21, tzinfo=timezone.utc)
FILTER_END   = datetime.now(timezone.utc)  # up to now


def backoff_sleep(attempt: int):
    base = min(60, (2 ** attempt))
    time.sleep(base * (0.5 + random.random() * 0.5))


def fetch_page(offset: int) -> Dict[str, Any]:
    payload = {
        "conversation_id": CONVERSATION_ID,
        "count": PAGE_SIZE,
        "offset": offset,
        "sort_by": "newest",
    }
    data = json.dumps(payload)
    attempt = 0
    while True:
        try:
            r = requests.post(API_URL, headers=HEADERS, data=data, timeout=30)
            if r.status_code in (429, 502, 503, 504):
                attempt += 1
                if attempt > MAX_RETRIES:
                    r.raise_for_status()
                backoff_sleep(attempt)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            attempt += 1
            if attempt > MAX_RETRIES:
                print(f"[fatal] giving up after {attempt} attempts at offset {offset}", file=sys.stderr)
                raise
            print(f"[warn] {e} — retrying (attempt {attempt})", file=sys.stderr)
            backoff_sleep(attempt)

def extract_sentiment(data):
    """
    Extract sentiment labels like BULLISH, BEARISH, NEUTRAL from the JSON.
    Returns a list of found sentiments (could be empty).
    """
    try:
        ids = data.get("additional_data", {}).get("labels", {}).get("ids", [])
        sentiments = [label.upper() for label in ids if label.upper() in {"BULLISH", "BEARISH", "NEUTRAL"}]
        return sentiments
    except Exception as e:
        print("Error parsing sentiment:", e)
        return []

def flatten_comment(c: Dict[str, Any]) -> Dict[str, Any]:
    reactions = c.get("reactions") or {}
    stats = c.get("stats") or {}
    content = c.get("content")

    message_text = None
    if isinstance(content, dict):
        message_text = content.get("text")
    elif isinstance(content, list):

        # Attempt to extract text from list elements if they are dicts with a 'text' key
        texts = [item.get('text') for item in content if isinstance(item, dict) and 'text' in item]
        message_text = " ".join(filter(None, texts)) # Join non-None texts


    return {
        "message_id": c.get("id"),
        "parent_id": c.get("parent_id"),
        "is_reply": c.get("parent_id") is not None,
        "created_at": datetime.fromtimestamp(c.get("written_at")).strftime('%Y-%m-%d'),
        "created_time":  datetime.fromtimestamp(c.get("time")).strftime('%H:%M:%S'),
        "author_id": c.get("user_id"),
        "user_reputation" : c.get("user_reputation"),
        "user_display_name" : c.get("user_display_name"),
        "message_text": message_text,
        "rank_score": c.get("rank_score"),
        "best_score": c.get("best_score"),
        "ranks_up": c.get("rank").get("ranks_up"),
        "ranks_down": c.get("rank").get("ranks_down"),
        "sentiment": extract_sentiment(c),
        "replies_count": c.get("total_replies_count"),
        "status": c.get("status")
    }


def write_jsonl(records: List[Dict[str, Any]], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path):
    if not records:
        path.write_text("", encoding="utf-8")
        return
    cols = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in records:
            w.writerow(r)


def parse_dt_safe(ts: str):
    if not ts:
        return None
    try:
        return dtparser.isoparse(ts).astimezone(timezone.utc)
    except Exception:
        return None


def main():
    print(f"Downloading Yahoo Finance TSLA forum via OpenWeb…")
    print(f"conversation_id: {CONVERSATION_ID}")
    print(f"Filtering messages from {FILTER_START.date()} to {FILTER_END.date()}")

    all_comments_raw: List[Dict[str, Any]] = []

    #last successfully downloaded comment in case of network disconnect or API problems then restart with the offset. Set to 1 to start downloading from the latest comment.
    offset = 302905

    data = fetch_page(offset)
    conv = data.get("conversation") or {}
    comments = conv.get("comments") or conv.get("messages") or []
    has_next = conv.get("has_next")
    next_offset = conv.get("offset") or (offset + len(comments))
    all_comments_raw.extend(comments)
    print(f"Fetched {len(comments)} (has_next={has_next})")

    #we download in batches of 10000 comments each. i indicates the batch number. it is necessary to keep the correct numeration of files for each batch. If network disconnects on batch 31 we restart from i=31 and rewrite the file
    for i in range(31, 42):
      while has_next and len(all_comments_raw)<10000:
          time.sleep(SLEEP_BETWEEN)
          data = fetch_page(next_offset)
          conv = data.get("conversation") or {}
          comments = conv.get("comments") or conv.get("messages") or []
          has_next = conv.get("has_next")
          next_offset = conv.get("offset") or (next_offset + len(comments))
          all_comments_raw.extend(comments)
          print(f"Fetched +{len(comments)} (total={len(all_comments_raw)}, has_next={has_next}), batch = {i}, next_offset={next_offset}")

      # Normalize + filter
      tidy = []
      for c in all_comments_raw:
          rec = flatten_comment(c)
          tidy.append(rec)
          dt = parse_dt_safe(rec["created_at"])


    # tidy.sort(key=lambda r: r["created_at"] or "")

      #Construct the filenames correctly using f-strings and Path objects
      write_jsonl(tidy, OUT_JSON.with_name(f"{OUT_JSON.name}_{i}.jsonl"))
      write_csv(tidy, OUT_CSV.with_name(f"{OUT_CSV.name}_{i}.csv"))

      print("\nDone:")
      print(f" - JSONL: {OUT_JSON.with_name(f'{OUT_JSON.name}_{i}.jsonl').resolve()}")
      print(f" -  CSV : {OUT_CSV.with_name(f'{OUT_CSV.name}_{i}.csv').resolve()}")
      print(f"Records in range: {len(tidy):,}")
      all_comments_raw = []

if __name__ == "__main__":
    main()
