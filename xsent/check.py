"""`python run_pipeline.py check` — verify API keys and connectivity before spending credits."""
from __future__ import annotations

import requests

from .config import Config


def check(cfg: Config) -> int:
    ok = True

    def line(name, good, msg, critical=True):
        nonlocal ok
        if critical:
            ok &= good
        tag = "OK  " if good else ("FAIL" if critical else "WARN")
        print(f"[{tag}] {name}: {msg}")

    # Apify
    if not cfg.apify_token:
        line("Apify", False, "APIFY_TOKEN missing in .env")
    else:
        try:
            r = requests.get("https://api.apify.com/v2/users/me",
                             headers={"Authorization": f"Bearer {cfg.apify_token}"}, timeout=20)
            if r.ok:
                d = r.json()["data"]
                plan = (d.get("plan") or {}).get("id", "?")
                line("Apify", True, f"user '{d.get('username')}', plan {plan}")
            else:
                line("Apify", False, f"HTTP {r.status_code} {r.text[:150]}")
            r = requests.get(f"https://api.apify.com/v2/acts/{cfg.apify_actor_id.replace('/', '~')}",
                             headers={"Authorization": f"Bearer {cfg.apify_token}"}, timeout=20)
            line("Apify actor", r.ok, cfg.apify_actor_id + ("" if r.ok else f" -> HTTP {r.status_code}"))
        except requests.RequestException as e:
            line("Apify", False, str(e))
    # Tavily
    if not cfg.tavily_api_key:
        line("Tavily", False, "TAVILY_API_KEY missing in .env")
    else:
        try:
            r = requests.post("https://api.tavily.com/search",
                              headers={"Authorization": f"Bearer {cfg.tavily_api_key}"},
                              json={"query": f"{cfg.asset_name} price", "topic": "news", "max_results": 1},
                              timeout=30)
            line("Tavily", r.ok, "search works" if r.ok else f"HTTP {r.status_code} {r.text[:150]}")
        except requests.RequestException as e:
            line("Tavily", False, str(e))
    # Prices
    try:
        import yfinance as yf
        h = yf.Ticker(cfg.price_ticker).history(period="5d")
        line("Prices (yfinance)", not h.empty, f"{cfg.price_ticker} last close {h['Close'].iloc[-1]:,.2f}"
             if not h.empty else "no data returned (Coinbase fallback will be used)", critical=False)
    except Exception as e:
        line("Prices (yfinance)", False, f"{e} (Coinbase fallback will be used)", critical=False)
    print("\nAll good - run: python run_pipeline.py all" if ok else "\nFix the FAIL lines above first.")
    return 0 if ok else 1
