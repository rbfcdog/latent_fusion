#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "live_news" / "capture.jsonl"
DST = ROOT / "data" / "live_news" / "capture_filtered.jsonl"

NOISE_PATTERNS = [
    r"\?",
    r"\b(should you|is it worth|why .* (plunged|surged|dropped|jumped)|3 reasons|5 reasons|top \d+|best \d+|worst \d+|here\'s why|stocks? to (buy|watch|sell)|is .* a (buy|sell)|can .* reach|what you need to know)\b",
    r"\b(how to|step-by-step|guide to|tips for|tricks)\b",
    r"\b(options whale alert|bearish short call|bullish call|unusual options activity)\b",
    r"\b(podcast|webinar|livestream|video)\b",
]
noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

MATERIAL_PATTERNS = [
    r"\b(earnings|quarterly|revenue|profit|guidance|q[1-4] (results|earnings|revenue)|beats?|misses?|sales decline|annual forecast|fiscal)\b",
    r"\b(acquir\w+|merger|buyout|takeover|divest\w+|joint venture)\b",
    r"\b(fda|phase [1-3]|clinical trial|efficacy|drug approval|patent|antitrust|sanctions?|probe|lawsuit|investigat\w+|sec probe|doj)\b",
    r"\b(ceo|cfo|appoints|resigns|layoffs?|restructur\w+|scraps? license|unveils? (new|ai|chip|model)|launches? (new|ai|chip)|tariffs?)\b",
    r"\b(dividend|share repurchase|buyback)\b",
]
material_re = re.compile("|".join(MATERIAL_PATTERNS), re.IGNORECASE)


def filter_material_news(src=SRC, dst=DST):
    total, passed = 0, 0
    by_ticker = Counter()
    with open(src, "r", encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            row = json.loads(line)
            title = row["title"]
            if noise_re.search(title):
                continue
            if material_re.search(title):
                passed += 1
                by_ticker[row["ticker"]] += 1
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "total": total, "passed": passed, "pct": round(passed / max(1, total) * 100, 2),
        "by_ticker": dict(by_ticker.most_common())
    }, indent=1))


if __name__ == "__main__":
    filter_material_news()
