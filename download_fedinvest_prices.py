from __future__ import annotations


import argparse
import datetime as dt
import io
import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


FEDINVEST_URL = "https://treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail"

# Light "browser-like" headers help avoid occasional blocking.
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "Origin": "https://treasurydirect.gov",
    "Referer": "https://treasurydirect.gov/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0 Safari/537.36"
    ),
}


def parse_tsy_price_to_decimal(x) -> pd._libs.missing.NAType | float:
    """
    Convert common Treasury price quote formats to decimal float.

    Handles:
      - "99.515625" (already decimal)
      - "99-16"     (99 + 16/32)
      - "99-16+"    (99 + (16.5)/32  i.e. + = 1/64)
      - "99-162"    (99 + (16 + 2/8)/32  last digit = 1/8 of 1/32 = 1/256)

    If it can't parse, returns pd.NA.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA

    # Already numeric?
    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)

    s = str(x).strip()
    if not s:
        return pd.NA

    # Try plain float first
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass

    # Try 32nds formats: 99-16, 99-16+, 99-162
    m = re.match(r"^(?P<pts>\d+)-(?P<frac>\d+)(?P<plus>\+)?$", s)
    if not m:
        return pd.NA

    pts = int(m.group("pts"))
    frac = m.group("frac")
    plus = m.group("plus") is not None

    # 2-digit: 32nds
    if len(frac) <= 2:
        thirty_seconds = int(frac)
        add = 0.5 if plus else 0.0  # '+' means half of 1/32
        return pts + (thirty_seconds + add) / 32.0

    # 3-digit: last digit is 1/8 of 1/32
    if len(frac) == 3 and frac.isdigit():
        thirty_seconds = int(frac[:2])
        eighths_of_32nd = int(frac[2])  # 0..7
        return pts + (thirty_seconds + eighths_of_32nd / 8.0) / 32.0

    return pd.NA


def _decode_response_content(resp: requests.Response) -> str:
    """
    The site is sometimes labeled ISO-8859-1; content is typically ASCII/UTF-8 compatible.
    We'll try UTF-8 first, then fall back.
    """
    try:
        return resp.content.decode("utf-8")
    except UnicodeDecodeError:
        return resp.content.decode("ISO-8859-1", errors="replace")


def fetch_fedinvest_prices(
    date: dt.date,
    session: Optional[requests.Session] = None,
    timeout: int = 30,
) -> pd.DataFrame:
    """
    Fetch CUSIP-level price data for a single date.
    Returns a DataFrame with columns:
      ['date','cusip','security_type','rate','maturity_date','call_date','bid','offer','eod_price']
    """
    payload = {
        "priceDateDay": str(date.day),
        "priceDateMonth": str(date.month),
        "priceDateYear": str(date.year),
        "fileType": "csv",
        "csv": "CSV FORMAT",
    }

    sess = session or requests.Session()
    resp = sess.post(FEDINVEST_URL, headers=HEADERS, data=payload, timeout=timeout)
    resp.raise_for_status()

    text = _decode_response_content(resp)
    if "<html" in text.lower():
        raise ValueError(f"Received HTML (not CSV) for {date}. Possibly no data or site changed.")

    df = pd.read_csv(io.StringIO(text), header=0)

    # The CSV is expected to have 8 columns (OpenBB renames them explicitly).
    # We'll defensively handle surprises.
    if df.shape[1] < 8:
        raise ValueError(f"Unexpected CSV shape for {date}: {df.shape}")

    df = df.iloc[:, :8].copy()
    df.columns = [
        "cusip",
        "security_type",
        "rate",
        "maturity_date",
        "call_date",
        "bid",
        "offer",
        "eod_price",
    ]
    df.insert(0, "date", pd.Timestamp(date))

    # Parse dates
    for c in ["maturity_date", "call_date"]:
        df[c] = pd.to_datetime(df[c], format="%m/%d/%Y", errors="coerce")

    # Rate (coupon) is numeric when present
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

    return df


def save_daily_frame(df: pd.DataFrame, out_path: Path) -> None:
    """
    Save as parquet if possible; otherwise save gzipped CSV.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Parquet is best for speed/size if pyarrow installed
        if out_path.suffix.lower() == ".parquet":
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False, compression="gzip")
    except Exception:
        # Fallback: gzip CSV
        fallback = out_path.with_suffix(".csv.gz")
        df.to_csv(fallback, index=False, compression="gzip")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download TreasuryDirect/FedInvest CUSIP-level daily prices."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument(
        "--outdir",
        default="fedinvest_prices",
        help="Output directory (default: fedinvest_prices)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv.gz"],
        default="parquet",
        help="Output format per day (default: parquet)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Seconds to sleep between requests (default: 0.25)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip dates already downloaded",
    )
    parser.add_argument(
        "--filter_notes_bonds",
        action="store_true",
        help="Keep only Note/Bond rows (drops Bills/TIPS/FRNs if present)",
    )
    parser.add_argument(
        "--parse_prices",
        action="store_true",
        help="Try to parse bid/offer/eod_price into decimal floats",
    )
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    outdir = Path(args.outdir)

    # Business days only (US holidays will be attempted and typically fail gracefully)
    dates = pd.bdate_range(start, end).date

    try:
        from tqdm import tqdm
        iterator = tqdm(dates, desc="Downloading")
    except Exception:
        iterator = dates

    sess = requests.Session()
    failures = []

    for d in iterator:
        fname = f"{d.isoformat()}.{args.format}"
        out_path = outdir / fname

        if args.resume and out_path.exists():
            continue

        try:
            df = fetch_fedinvest_prices(d, session=sess)

            if args.filter_notes_bonds:
                mask = df["security_type"].astype(str).str.contains("note|bond", case=False, na=False)
                df = df.loc[mask].copy()

            if args.parse_prices:
                for c in ["bid", "offer", "eod_price"]:
                    df[c] = df[c].apply(parse_tsy_price_to_decimal)

            save_daily_frame(df, out_path)

        except Exception as e:
            failures.append((d.isoformat(), str(e)))

        time.sleep(args.sleep)

    if failures:
        fail_path = outdir / "_failures.csv"
        pd.DataFrame(failures, columns=["date", "error"]).to_csv(fail_path, index=False)
        print(f"\nDone with {len(failures)} failures. See: {fail_path}")
    else:
        print("\nDone with zero failures.")


if __name__ == "__main__":
    main()