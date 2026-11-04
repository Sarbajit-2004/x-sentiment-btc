"""Sanity-check that a historical tweet CSV carries its real dates before using it for training.

Three independent checks, each cheap to stream over a multi-GB file:
  1. newest account-creation date (no account can be newer than the tweets' real era);
  2. which years the tweet text itself mentions;
  3. weekday words ("happy Monday", "today is Friday", "#Thursday") against the stamped date's weekday.

On genuine data check 3 agrees for most tweets. On the Kaggle "Bitcoin Tweets Dataset (2025-2026)"
(pokeash) it agrees for under 1% and the text mostly mentions 2021-22: the tweets were re-dated.

    python scripts/validate_tweet_dates.py path/to/tweets.csv
"""
from __future__ import annotations

import re
import sys

import pandas as pd

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
YEAR = re.compile(r"\b(20(?:1[5-9]|2[0-9]))\b")
WEEKDAY = re.compile(r"(?:happy |today is |#)(" + "|".join(DAYS) + r")\b")


def main(path: str, date_col: str = "date", text_col: str = "text", created_col: str = "user_created") -> int:
    months, years = {}, {}
    newest_account, rows, match, mismatch = None, 0, 0, 0
    reader = pd.read_csv(path, chunksize=500_000, dtype=str, on_bad_lines="skip",
                         encoding_errors="ignore", escapechar="\\")
    for ch in reader:
        rows += len(ch)
        d = pd.to_datetime(ch[date_col], errors="coerce", utc=True, format="mixed")
        for k, v in d.dt.strftime("%Y-%m").value_counts().items():
            months[k] = months.get(k, 0) + v
        if created_col in ch:
            c = pd.to_datetime(ch[created_col], errors="coerce", format="mixed").max()
            newest_account = c if newest_account is None or (pd.notna(c) and c > newest_account) else newest_account
        txt = ch[text_col].fillna("").str.lower()
        for ys in txt.str.findall(YEAR):
            for y in ys:
                years[y] = years.get(y, 0) + 1
        for t, dd in zip(txt, d):
            m = WEEKDAY.search(t)
            if m and pd.notna(dd):
                if DAYS.index(m.group(1)) == dd.day_of_week:
                    match += 1
                else:
                    mismatch += 1
    print(f"rows: {rows:,}")
    print("stamped months:", ", ".join(f"{k}: {v:,}" for k, v in sorted(months.items())))
    print(f"newest account created: {newest_account}")
    print("years mentioned in text:", ", ".join(f"{k}: {v:,}" for k, v in sorted(years.items())))
    tot = match + mismatch
    print(f"weekday words agreeing with stamped date: {match:,} of {tot:,}"
          + (f" ({match / tot:.1%}; ~14% would be chance)" if tot else ""))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(*sys.argv[1:]))
