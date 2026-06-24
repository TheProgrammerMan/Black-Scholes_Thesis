from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ============================================================
# SETTINGS
# ============================================================
BASE_URL   = "https://api.massive.com"
API_KEY    = "WaiN7WF3QsDH2Ptk_RiFdRGEFDgyxoSi"
UNDERLYING = "SPX"
OUT_CSV    = "SPX_ATM_calls_MayJul2024.csv"

MIN_SECONDS_BETWEEN_REQUESTS = 5
MAX_RETRIES = 6
TIMEOUT_S   = 60   # increased from 30

# ============================================================
# INPUT DATA
# ============================================================
SPX_CLOSES = [
    {"date": "2024-05-27", "Close": 5277.51},
    {"date": "2024-06-03", "Close": 5346.99},
    {"date": "2024-06-10", "Close": 5431.60},
    {"date": "2024-06-17", "Close": 5464.62},
    {"date": "2024-06-24", "Close": 5460.48},
    {"date": "2024-07-01", "Close": 5567.19},
    {"date": "2024-07-08", "Close": 5615.35},
    {"date": "2024-07-15", "Close": 5505.00},
    {"date": "2024-07-22", "Close": 5459.10},
    {"date": "2024-07-29", "Close": 5436.44},
]

# ============================================================
# RATE LIMITER
# ============================================================
class RateLimiter:
    def __init__(self, min_interval_s: float):
        self.min_interval_s = float(min_interval_s)
        self._last_ts = 0.0

    def wait(self):
        elapsed = time.time() - self._last_ts
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)
        self._last_ts = time.time()


# ============================================================
# HTTP SESSION
# ============================================================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
    })
    return s


def get_json(session: requests.Session, limiter: RateLimiter, url: str) -> Dict[str, Any]:
    last_err: Optional[Exception] = None
    for i in range(MAX_RETRIES):
        try:
            limiter.wait()
            print(f"    -> GET {url[:80]}...")
            r = session.get(url, timeout=TIMEOUT_S)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"  429 rate-limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            backoff = (2 ** i) * 10.0   # longer back-off: 10s, 20s, 40s ...
            print(f"  Attempt {i+1} failed ({e}), retrying in {backoff:.0f}s ...")
            time.sleep(backoff)
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries. Last error: {last_err}")


# ============================================================
# FETCH ALL CALL CONTRACTS FOR A GIVEN DATE (paginated)
# ============================================================
def fetch_call_contracts(session: requests.Session, limiter: RateLimiter, as_of: str, spot: float) -> pd.DataFrame:
    # Filter to only the ATM-neighbourhood strikes (spot +/- 5%) to dramatically
    # reduce the number of pages returned and avoid mid-pagination timeouts.
    # Use strike_price filters (±3% of spot) so we only fetch the small slice
    # of contracts near the money — avoids pulling 12,000+ rows across 13 pages.
    low  = round(spot * 0.97 / 5) * 5   # round to nearest $5 strike
    high = round(spot * 1.03 / 5) * 5
    url = (
        f"{BASE_URL}/v3/reference/options/contracts"
        f"?underlying_ticker={UNDERLYING}"
        f"&contract_type=call"
        f"&as_of={as_of}"
        f"&strike_price.gte={low}"
        f"&strike_price.lte={high}"
        f"&limit=1000"
    )
    rows: List[Dict[str, Any]] = []
    page = 0
    while url:
        page += 1
        data = get_json(session, limiter, url)
        status = data.get("status")
        if status not in ("OK", "DELAYED"):
            raise RuntimeError(f"Bad API status for as_of={as_of}: {data}")
        rows.extend(data.get("results", []))
        url = data.get("next_url")
        print(f"    Contracts page {page}: {len(rows)} total so far")

    if not rows:
        raise RuntimeError(f"No contracts returned for as_of={as_of}.")
    return pd.DataFrame(rows)


# ============================================================
# PICK ATM STRIKE PER EXPIRATION (capped at 1 year out)
# ============================================================
def select_atm_contracts(contracts_df: pd.DataFrame, spot: float, as_of: str) -> pd.DataFrame:
    df = contracts_df.copy()
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df = df.dropna(subset=["expiration_date", "strike_price", "ticker"])

    # Keep only expirations between as_of and exactly 1 month out
    max_exp = (pd.Timestamp(as_of) + pd.DateOffset(months=1)).date().isoformat()
    df = df[(df["expiration_date"] >= as_of) & (df["expiration_date"] <= max_exp)]

    # Find the single ATM strike closest to spot across ALL contracts
    strikes_all = df["strike_price"].to_numpy()
    if strikes_all.size == 0:
        return pd.DataFrame()
    atm_strike = float(strikes_all[(abs(strikes_all - spot)).argmin()])

    # Keep only contracts at that ATM strike, one row per expiration
    atm_df = (
        df[df["strike_price"] == atm_strike]
        .drop_duplicates(subset=["expiration_date"])
        .sort_values("expiration_date")
    )

    atm_df = atm_df[["expiration_date", "strike_price", "ticker"]].copy()
    atm_df.insert(0, "date", as_of)
    atm_df.insert(2, "spot_price", spot)
    atm_df = atm_df.reset_index(drop=True)
    return atm_df


# ============================================================
# FETCH CLOSE PRICE FOR ONE CONTRACT ON ONE DATE
# ============================================================
def fetch_close(
    session: requests.Session,
    limiter: RateLimiter,
    option_ticker: str,
    trade_date: str,
) -> Optional[float]:
    end_plus_1 = (pd.Timestamp(trade_date).date() + timedelta(days=1)).isoformat()
    url = (
        f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day"
        f"/{trade_date}/{end_plus_1}"
        f"?adjusted=true&sort=asc&limit=1"
    )
    data    = get_json(session, limiter, url)
    results = data.get("results", [])
    return float(results[0]["c"]) if results else None


# ============================================================
# DIAGNOSTIC
# ============================================================
def diagnose(session: requests.Session) -> None:
    """Fire a minimal known-good request and print the raw response so we
    can confirm auth, connectivity, and plan access before the main run."""
    test_url = f"{BASE_URL}/v3/reference/options/contracts?underlying_ticker=SPX&contract_type=call&limit=1"
    print("=== DIAGNOSTIC ===")
    print(f"  URL: {test_url}")
    try:
        r = session.get(test_url, timeout=TIMEOUT_S)
        print(f"  HTTP status : {r.status_code}")
        print(f"  Response    : {r.text[:500]}")
    except Exception as e:
        print(f"  FAILED      : {e}")
    print("==================\n")


# ============================================================
# MAIN
# ============================================================
def main():
    session = make_session()
    limiter = RateLimiter(MIN_SECONDS_BETWEEN_REQUESTS)

    diagnose(session)

    spx_df = pd.DataFrame(SPX_CLOSES)
    spx_df["date"] = pd.to_datetime(spx_df["date"]).dt.date

    all_rows: List[Dict[str, Any]] = []

    for _, row in spx_df.iterrows():
        as_of = row["date"].isoformat()
        spot  = float(row["Close"])
        print(f"\n--- Processing {as_of} (spot={spot}) ---")

        contracts = fetch_call_contracts(session, limiter, as_of, spot)
        atm       = select_atm_contracts(contracts, spot, as_of)
        print(f"  ATM contracts selected: {len(atm)}")

        for i, (_, opt) in enumerate(atm.iterrows(), start=1):
            ticker    = opt["ticker"]
            close     = fetch_close(session, limiter, ticker, as_of)
            moneyness = spot / opt["strike_price"] if opt["strike_price"] else None

            all_rows.append({
                "date":            as_of,
                "expiration_date": opt["expiration_date"],
                "spot_price":      spot,
                "strike_price":    opt["strike_price"],
                "ticker":          ticker,
                "close":           close,
                "moneyness_close": round(moneyness, 6) if moneyness else None,
            })

            if i % 20 == 0:
                print(f"    ... fetched {i}/{len(atm)} contracts for {as_of}")

        print(f"  Done {as_of}. Running total rows: {len(all_rows)}")

    result = pd.DataFrame(all_rows)
    # Normalise both date columns to YYYY-MM-DD regardless of how the API returned them
    for col in ("date", "expiration_date"):
        result[col] = pd.to_datetime(result[col], dayfirst=True).dt.strftime("%Y-%m-%d")
    result.sort_values(["date", "expiration_date"], inplace=True)
    result.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(result):,} rows, {len(result.columns)} columns)")


if __name__ == "__main__":
    main()
