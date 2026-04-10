from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


# -----------------------------
# Config / Constants
# -----------------------------

FEDINVEST_URL = "https://treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail"

FEDINVEST_HEADERS = {
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

# Stooq CSV download endpoint pattern is widely used; supports optional d1/d2 in YYYYMMDD.
# Example patterns appear in public docs/issues: .../q/d/l/?s=tsla.us&i=d and with d1/d2.
STOOQ_URL = "https://stooq.com/q/d/l/"

# CME month codes (we only need quarterly for ZN): H=Mar, M=Jun, U=Sep, Z=Dec
MONTH_CODE = {3: "H", 6: "M", 9: "U", 12: "Z"}

US_BDAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


@dataclass(frozen=True)
class ZNContract:
    year: int
    month: int  # 3,6,9,12
    symbol: str  # e.g. znh25.f
    delivery_month_start: dt.date  # calendar first day
    last_trading_day: dt.date  # "day prior to last 7 business days" (approx w/ US fed holidays)


# -----------------------------
# Helpers: dates / calendars
# -----------------------------

def to_yyyymmdd(d: dt.date) -> str:
    return d.strftime("%Y%m%d")


def month_last_calendar_day(year: int, month: int) -> dt.date:
    if month == 12:
        return dt.date(year, 12, 31)
    return dt.date(year, month + 1, 1) - dt.timedelta(days=1)


def business_days_in_month(year: int, month: int) -> List[dt.date]:
    start = pd.Timestamp(dt.date(year, month, 1))
    end = pd.Timestamp(month_last_calendar_day(year, month))
    bdays = pd.date_range(start, end, freq=US_BDAY)
    return [d.date() for d in bdays]


def first_business_day_on_or_after(d: dt.date) -> dt.date:
    ts = pd.Timestamp(d)
    if ts.weekday() >= 5 or ts in USFederalHolidayCalendar().holidays(ts, ts):
        # Move forward to next business day
        ts2 = ts
        while True:
            ts2 = ts2 + pd.Timedelta(days=1)
            if ts2.weekday() < 5 and ts2 not in USFederalHolidayCalendar().holidays(ts2, ts2):
                return ts2.date()
    return d


def last_business_day_of_month(year: int, month: int) -> dt.date:
    bdays = business_days_in_month(year, month)
    return bdays[-1]


def last_trading_day_zn(year: int, month: int) -> dt.date:
    """
    ZN last trading day rule (approx): "day prior to last seven business days of contract month".
    So if bdays = [ ... ], last 7 are bdays[-7:]; day prior is bdays[-8].
    """
    bdays = business_days_in_month(year, month)
    if len(bdays) < 8:
        raise ValueError("Unexpectedly few business days in month.")
    return bdays[-8]


def generate_quarterly_contracts(start: dt.date, end: dt.date) -> List[ZNContract]:
    """
    Generate quarterly ZN contracts covering [start, end], plus one extra quarter for safety.
    """
    # Find first quarterly month >= start month (or next quarter)
    y, m = start.year, start.month
    quarters = [3, 6, 9, 12]

    def next_quarter(y0: int, m0: int) -> Tuple[int, int]:
        for q in quarters:
            if m0 <= q:
                return y0, q
        return y0 + 1, 3

    cy, cm = next_quarter(y, m)
    contracts: List[ZNContract] = []

    # build until after end (one extra quarter)
    while True:
        code = MONTH_CODE[cm]
        yy = str(cy)[-2:]
        sym = f"zn{code.lower()}{yy}.f"  # stooq format seems to accept lower-case
        dstart = dt.date(cy, cm, 1)
        ltd = last_trading_day_zn(cy, cm)
        contracts.append(ZNContract(cy, cm, sym, dstart, ltd))

        # stop condition
        if dt.date(cy, cm, 1) > (end + relativedelta(months=3)):
            break

        # advance quarter
        if cm == 12:
            cy += 1
            cm = 3
        else:
            cm += 3

    return contracts


def active_contract_for_date(d: dt.date, contracts: List[ZNContract]) -> ZNContract:
    for c in contracts:
        if d <= c.last_trading_day:
            return c
    return contracts[-1]


# -----------------------------
# Pricing parsing
# -----------------------------

def parse_tsy_price_to_decimal(x) -> pd._libs.missing.NAType | float:
    """
    Convert common Treasury quote formats to decimal float.
    Handles:
      - 99.515625
      - 99-16
      - 99-16+
      - 99-162 (1/8 of 1/32)
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return pd.NA

    if isinstance(x, (int, float)) and not pd.isna(x):
        return float(x)

    s = str(x).strip()
    if not s:
        return pd.NA

    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass

    m = re.match(r"^(?P<pts>\d+)-(?P<frac>\d+)(?P<plus>\+)?$", s)
    if not m:
        return pd.NA

    pts = int(m.group("pts"))
    frac = m.group("frac")
    plus = m.group("plus") is not None

    if len(frac) <= 2:
        th = int(frac)
        add = 0.5 if plus else 0.0
        return pts + (th + add) / 32.0

    if len(frac) == 3 and frac.isdigit():
        th = int(frac[:2])
        eighth = int(frac[2])
        return pts + (th + eighth / 8.0) / 32.0

    return pd.NA


# -----------------------------
# TreasuryDirect / FedInvest client (approach 1)
# -----------------------------

class FedInvestClient:
    def __init__(self, cache_dir: Path, sleep: float = 0.25):
        self.cache_dir = cache_dir
        self.sleep = sleep
        self.session = requests.Session()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, date: dt.date) -> Path:
        return self.cache_dir / f"fedinvest_{date.isoformat()}.parquet"

    @staticmethod
    def _decode(resp: requests.Response) -> str:
        try:
            return resp.content.decode("utf-8")
        except UnicodeDecodeError:
            return resp.content.decode("ISO-8859-1", errors="replace")

    def fetch_day(self, date: dt.date) -> pd.DataFrame:
        payload = {
            "priceDateDay": str(date.day),
            "priceDateMonth": str(date.month),
            "priceDateYear": str(date.year),
            "fileType": "csv",
            "csv": "CSV FORMAT",
        }
        r = self.session.post(FEDINVEST_URL, headers=FEDINVEST_HEADERS, data=payload, timeout=30)
        r.raise_for_status()
        text = self._decode(r)
        if "<html" in text.lower():
            raise ValueError(f"Got HTML (not CSV) for {date} — likely no data/holiday.")

        df = pd.read_csv(io.StringIO(text), header=0)
        if df.shape[1] < 8:
            raise ValueError(f"Unexpected FedInvest CSV shape {df.shape} on {date}")

        df = df.iloc[:, :8].copy()
        df.columns = [
            "cusip",
            "security_type",
            "coupon",
            "maturity_date",
            "call_date",
            "bid",
            "offer",
            "eod_price",
        ]
        df.insert(0, "date", pd.Timestamp(date))

        # Parse dates
        df["maturity_date"] = pd.to_datetime(df["maturity_date"], format="%m/%d/%Y", errors="coerce")
        df["call_date"] = pd.to_datetime(df["call_date"], format="%m/%d/%Y", errors="coerce")

        # Coupon as numeric (% -> decimal)
        df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce") / 100.0

        # Parse prices
        for c in ["bid", "offer", "eod_price"]:
            df[c] = df[c].apply(parse_tsy_price_to_decimal)

        # Pick a "clean_price" field: mid if bid/offer exist, else eod
        bid = df["bid"]
        off = df["offer"]
        mid = (bid + off) / 2.0
        df["clean_price"] = mid.where(~mid.isna(), df["eod_price"])

        time.sleep(self.sleep)
        return df

    def get_day(self, date: dt.date, use_cache: bool = True) -> Optional[pd.DataFrame]:
        path = self._cache_path(date)
        if use_cache and path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

        try:
            df = self.fetch_day(date)
        except Exception:
            return None

        try:
            df.to_parquet(path, index=False)
        except Exception:
            # fallback
            df.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")
        return df


# -----------------------------
# Stooq futures downloader
# -----------------------------

class StooqClient:
    def __init__(self, cache_dir: Path, sleep: float = 0.2):
        self.cache_dir = cache_dir
        self.sleep = sleep
        self.session = requests.Session()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, symbol: str, start: dt.date, end: dt.date) -> Path:
        safe = symbol.replace(".", "_")
        return self.cache_dir / f"stooq_{safe}_{start.isoformat()}_{end.isoformat()}.parquet"

    def fetch_symbol(self, symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        params = {
            "s": symbol,
            "i": "d",
            "d1": to_yyyymmdd(start),
            "d2": to_yyyymmdd(end),
        }
        r = self.session.get(STOOQ_URL, params=params, timeout=30)
        r.raise_for_status()

        # Stooq returns CSV with header: Date,Open,High,Low,Close,Volume
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or "Date" not in df.columns:
            raise ValueError(f"No data for {symbol} in range.")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df = df.sort_values("date")
        time.sleep(self.sleep)
        return df

    def get_symbol(self, symbol: str, start: dt.date, end: dt.date, use_cache: bool = True) -> pd.DataFrame:
        path = self._cache_path(symbol, start, end)
        if use_cache and path.exists():
            return pd.read_parquet(path)

        df = self.fetch_symbol(symbol, start, end)
        try:
            df.to_parquet(path, index=False)
        except Exception:
            df.to_csv(path.with_suffix(".csv.gz"), index=False, compression="gzip")
        return df


# -----------------------------
# Repo rate helpers (optional)
# -----------------------------

def fetch_repo_series(cache_dir: Path) -> pd.Series:
    """
    Returns a daily repo proxy series (percent):
      - Primary Dealer Survey GC Repo Rate through 2018-02-28 (NY Fed xlsx)
      - SOFR from FRED afterward
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) PD Survey file (xlsx)
    pd_path = cache_dir / "HistoricalOvernightTreasGCRepoPriDealerSurvRate.xlsx"
    if not pd_path.exists():
        url = "https://www.newyorkfed.org/medialibrary/media/markets/HistoricalOvernightTreasGCRepoPriDealerSurvRate.xlsx"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        pd_path.write_bytes(r.content)

    pd_df = pd.read_excel(pd_path, sheet_name=0)
    pd_df.columns = ["date", "repo"]
    pd_df["date"] = pd.to_datetime(pd_df["date"])
    pd_df["repo"] = pd.to_numeric(pd_df["repo"], errors="coerce")
    pd_repo = pd_df.set_index("date")["repo"]

    # 2) SOFR from FRED CSV
    # Works without API key in most environments.
    sofr_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
    r = requests.get(sofr_url, timeout=30)
    r.raise_for_status()
    sofr_df = pd.read_csv(io.StringIO(r.text))
    sofr_df.columns = ["date", "repo"]
    sofr_df["date"] = pd.to_datetime(sofr_df["date"])
    sofr_df["repo"] = pd.to_numeric(sofr_df["repo"], errors="coerce")
    sofr_repo = sofr_df.set_index("date")["repo"]

    repo = pd.concat([pd_repo, sofr_repo]).sort_index()
    return repo


# -----------------------------
# Conversion factor (CME formula)
# -----------------------------

def round_coupon_to_nearest_1_8_percent(coupon_decimal: float) -> float:
    """
    CME: coupon in decimals rounded to nearest one-eighth of one percent (ties rounded up).
    1/8 of 1% = 0.125% = 0.00125 in decimal.
    """
    step = 0.00125
    x = coupon_decimal / step
    flo = math.floor(x)
    frac = x - flo
    if frac > 0.5:
        return (flo + 1) * step
    if frac < 0.5:
        return flo * step
    # tie
    return (flo + 1) * step


def conversion_factor_zn(coupon_decimal: float, maturity: dt.date, delivery_month_start: dt.date) -> float:
    """
    Compute ZN conversion factor using CME's published formula (6% standard, 3-month rounding).
    Returns factor rounded to 4 decimals.
    """
    coupon = round_coupon_to_nearest_1_8_percent(coupon_decimal)

    rd = relativedelta(maturity, delivery_month_start)
    n = rd.years
    months = rd.months

    # ZN rounds z down to nearest quarter-year increment in months: 0,3,6,9
    z = (months // 3) * 3

    # helper v (ZN uses 3-month convention; if z >= 7, v=3)
    if z < 7:
        v = z
        c = 1 / (1.03 ** (2 * n))
    else:
        v = 3
        c = 1 / (1.03 ** (2 * n + 1))

    a = 1 / (1.03 ** (v / 6))
    b = (coupon / 2) * (6 - v) / 6
    d = (coupon / 0.06) * (1 - c)

    factor = a * ((coupon / 2) + c + d) - b
    return round(factor, 4)


# -----------------------------
# Coupon schedule + accrued interest
# -----------------------------

def last_next_coupon_dates(maturity: dt.date, asof: dt.date) -> Tuple[dt.date, dt.date]:
    """
    For a Treasury with semiannual coupons on maturity day-of-month,
    walk backward from maturity in 6M steps to find last <= asof < next.
    """
    nxt = maturity
    while True:
        prev = nxt - relativedelta(months=6)
        if prev <= asof < nxt:
            return prev, nxt
        nxt = prev


def accrued_interest(coupon_decimal: float, maturity: dt.date, asof: dt.date) -> float:
    last_cpn, next_cpn = last_next_coupon_dates(maturity, asof)
    period_days = (next_cpn - last_cpn).days
    if period_days <= 0:
        return 0.0
    accr_days = (asof - last_cpn).days
    accr_days = max(accr_days, 0)
    return (coupon_decimal / 2) * 100.0 * (accr_days / period_days)


def coupon_cashflows_between(coupon_decimal: float, maturity: dt.date, start_excl: dt.date, end_incl: dt.date) -> float:
    """
    Sum coupons paid in (start_excl, end_incl].
    """
    amt = (coupon_decimal / 2) * 100.0
    total = 0.0
    d = maturity
    # walk backwards
    while d > start_excl:
        if start_excl < d <= end_incl:
            total += amt
        d = d - relativedelta(months=6)
    return total


# -----------------------------
# Basket + CTD calculations
# -----------------------------

def build_zn_deliverable_basket(
    contract: ZNContract,
    fed: FedInvestClient,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build deliverable basket as-of a reference business day in the delivery month.
    Eligibility window (ZN): remaining term to maturity >= 6y6m and < 8y from 1st day of delivery month.
    """
    ref = first_business_day_on_or_after(contract.delivery_month_start)
    day_df = fed.get_day(ref, use_cache=use_cache)
    if day_df is None or day_df.empty:
        raise ValueError(f"Could not fetch FedInvest data for basket reference date {ref}")

    df = day_df.copy()
    df = df.dropna(subset=["cusip", "coupon", "maturity_date", "clean_price"])
    df["maturity_date"] = pd.to_datetime(df["maturity_date"]).dt.date

    # filter to (nominal) notes
    st = df["security_type"].astype(str).str.lower()
    is_note = st.str.contains("note", na=False)
    not_infl = ~st.str.contains("tips|inflation", na=False)
    not_frn = ~st.str.contains("frn|floating", na=False)
    not_bill = ~st.str.contains("bill", na=False)
    df = df[is_note & not_infl & not_frn & not_bill].copy()

    start = contract.delivery_month_start
    lo = start + relativedelta(years=6, months=6)
    hi = start + relativedelta(years=8)

    df = df[(df["maturity_date"] >= lo) & (df["maturity_date"] < hi)].copy()

    if df.empty:
        raise ValueError(f"No deliverables found for {contract.symbol} (check filters).")

    # conversion factor
    df["cf"] = df.apply(lambda r: conversion_factor_zn(float(r["coupon"]), r["maturity_date"], start), axis=1)

    # keep essential columns
    out = df[["cusip", "coupon", "maturity_date", "cf"]].drop_duplicates().reset_index(drop=True)
    return out


def implied_repo_rate(
    clean_price: float,
    coupon_decimal: float,
    maturity: dt.date,
    futures_price: float,
    cf: float,
    val_date: dt.date,
    delivery_date: dt.date,
) -> Tuple[float, float, float, float]:
    """
    Simple implied repo (IRR) approximation:
      dirty_now = clean + AI(now)
      invoice   = futures_price * cf + AI(delivery)
      coupons   = sum coupons between (val_date, delivery_date]
      irr       = ((invoice + coupons) / dirty_now - 1) * 360/days
    Returns: (irr_percent, dirty_now, invoice, coupons)
    """
    ai_now = accrued_interest(coupon_decimal, maturity, val_date)
    dirty = clean_price + ai_now

    ai_del = accrued_interest(coupon_decimal, maturity, delivery_date)
    invoice = futures_price * cf + ai_del

    coupons = coupon_cashflows_between(coupon_decimal, maturity, val_date, delivery_date)
    days = (delivery_date - val_date).days
    if days <= 0 or dirty <= 0:
        return (float("nan"), dirty, invoice, coupons)

    irr = ((invoice + coupons) / dirty - 1.0) * (360.0 / days)
    return (irr * 100.0, dirty, invoice, coupons)


def compute_daily_ctd(
    date: dt.date,
    contract: ZNContract,
    futures_px: float,
    basket: pd.DataFrame,
    cash_day: pd.DataFrame,
    delivery_date: dt.date,
    repo_series: Optional[pd.Series] = None,
) -> Optional[dict]:
    """
    Compute CTD as max implied repo among basket constituents.
    """
    cash = cash_day[["cusip", "clean_price", "coupon", "maturity_date"]].copy()
    cash = cash.dropna(subset=["cusip", "clean_price", "coupon", "maturity_date"])
    cash["maturity_date"] = pd.to_datetime(cash["maturity_date"]).dt.date

    merged = basket.merge(cash, on="cusip", how="inner", suffixes=("", "_day"))
    if merged.empty:
        return None

    rows = []
    for r in merged.itertuples(index=False):
        irr, dirty, invoice, coupons = implied_repo_rate(
            clean_price=float(r.clean_price),
            coupon_decimal=float(r.coupon),
            maturity=r.maturity_date,
            futures_price=float(futures_px),
            cf=float(r.cf),
            val_date=date,
            delivery_date=delivery_date,
        )
        gross_basis = float(r.clean_price) - float(futures_px) * float(r.cf)
        rows.append(
            {
                "cusip": r.cusip,
                "coupon": float(r.coupon),
                "maturity": r.maturity_date,
                "cf": float(r.cf),
                "clean_price": float(r.clean_price),
                "dirty_price": float(dirty),
                "invoice_price": float(invoice),
                "coupons_to_delivery": float(coupons),
                "gross_basis": float(gross_basis),
                "implied_repo_pct": float(irr),
            }
        )

    df = pd.DataFrame(rows).dropna(subset=["implied_repo_pct"])
    if df.empty:
        return None

    df = df.sort_values("implied_repo_pct", ascending=False)
    ctd = df.iloc[0]

    repo = float("nan")
    if repo_series is not None:
        ts = pd.Timestamp(date)
        if ts in repo_series.index:
            repo = float(repo_series.loc[ts])
        else:
            # try last available (ffill style)
            prior = repo_series.loc[:ts].dropna()
            if not prior.empty:
                repo = float(prior.iloc[-1])

    out = {
        "date": date,
        "contract": contract.symbol,
        "delivery_month": contract.delivery_month_start.strftime("%Y-%m"),
        "futures_price": float(futures_px),
        "delivery_date_used": delivery_date,
        "ctd_cusip": ctd["cusip"],
        "ctd_maturity": ctd["maturity"],
        "ctd_coupon": ctd["coupon"],
        "ctd_cf": ctd["cf"],
        "ctd_clean_price": ctd["clean_price"],
        "ctd_dirty_price": ctd["dirty_price"],
        "ctd_invoice_price": ctd["invoice_price"],
        "ctd_coupons_to_delivery": ctd["coupons_to_delivery"],
        "ctd_gross_basis": ctd["gross_basis"],
        "ctd_implied_repo_pct": ctd["implied_repo_pct"],
        "repo_proxy_pct": repo,
        "ctd_implied_minus_repo_pct": (ctd["implied_repo_pct"] - repo) if pd.notna(repo) else float("nan"),
        "basket_size_seen": int(len(df)),
        # (optional) keep runner-up too
        "ctd2_cusip": df.iloc[1]["cusip"] if len(df) > 1 else None,
        "ctd2_implied_repo_pct": float(df.iloc[1]["implied_repo_pct"]) if len(df) > 1 else float("nan"),
    }
    return out


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--out", default="zn_ctd_2015_2025.csv")
    p.add_argument("--cache", default="cache_zn_basis")
    p.add_argument("--sleep_fedinvest", type=float, default=0.25)
    p.add_argument("--sleep_stooq", type=float, default=0.2)
    p.add_argument("--no_repo", action="store_true", help="Skip repo proxy download (faster).")
    args = p.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    cache_dir = Path(args.cache)
    fed = FedInvestClient(cache_dir / "fedinvest", sleep=args.sleep_fedinvest)
    stooq = StooqClient(cache_dir / "stooq", sleep=args.sleep_stooq)

    contracts = generate_quarterly_contracts(start, end)
    print(f"Generated {len(contracts)} quarterly ZN contracts.")

    # Repo (optional)
    repo_series = None
    if not args.no_repo:
        try:
            repo_series = fetch_repo_series(cache_dir / "repo")
            print("Loaded repo proxy series (PD survey then SOFR).")
        except Exception as e:
            print(f"Repo proxy download failed (continuing without repo): {e}")
            repo_series = None

    # Download futures data for each contract once
    futures_by_symbol: Dict[str, pd.DataFrame] = {}
    for c in contracts:
        try:
            df = stooq.get_symbol(c.symbol, start - relativedelta(months=2), end + relativedelta(months=2))
            futures_by_symbol[c.symbol] = df
        except Exception as e:
            print(f"WARNING: could not fetch futures for {c.symbol}: {e}")

    # Pre-build deliverable baskets per contract (cached in-memory)
    baskets: Dict[str, pd.DataFrame] = {}
    for c in contracts:
        try:
            baskets[c.symbol] = build_zn_deliverable_basket(c, fed, use_cache=True)
        except Exception as e:
            print(f"WARNING: could not build basket for {c.symbol}: {e}")

    # Iterate business days in [start, end]
    bdays = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=US_BDAY)
    results: List[dict] = []

    try:
        from tqdm import tqdm
        iterator: Iterable[pd.Timestamp] = tqdm(bdays, desc="CTD")
    except Exception:
        iterator = bdays

    for ts in iterator:
        d = ts.date()
        c = active_contract_for_date(d, contracts)

        if c.symbol not in futures_by_symbol or c.symbol not in baskets:
            continue

        fut_df = futures_by_symbol[c.symbol]
        # match date
        row = fut_df.loc[fut_df["date"] == pd.Timestamp(d)]
        if row.empty:
            continue
        fut_px = float(row.iloc[0]["close"])

        cash_day = fed.get_day(d, use_cache=True)
        if cash_day is None or cash_day.empty:
            continue

        # choose delivery date assumption: last business day of delivery month
        deliv_date = last_business_day_of_month(c.year, c.month)

        out = compute_daily_ctd(
            date=d,
            contract=c,
            futures_px=fut_px,
            basket=baskets[c.symbol],
            cash_day=cash_day,
            delivery_date=deliv_date,
            repo_series=repo_series,
        )
        if out is not None:
            results.append(out)

    out_df = pd.DataFrame(results).sort_values("date")
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df):,} rows to: {out_path.resolve()}")


if __name__ == "__main__":
    main()