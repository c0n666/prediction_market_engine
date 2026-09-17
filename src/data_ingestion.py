"""
Asynchronous Polymarket REST ingestion, historical persistence, and synthetic
time-series generation for offline experiments / practice backtests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd

from config import config, ensure_data_dirs

logger = logging.getLogger(__name__)

# User-facing category → Polymarket Gamma `tag_slug` values
CATEGORY_TAG_MAP: Dict[str, List[str]] = {
    "Macro": ["macro", "economy", "fed", "inflation", "cpi", "finance"],
    "Crypto": ["crypto", "bitcoin", "ethereum", "btc", "eth"],
    "Politics": ["politics", "elections", "trump", "usa-elections"],
    "Other": [],
}

# Niche categories need a wider scan to surface enough markets
NICHE_CATEGORIES = frozenset({"Macro"})
NICHE_FETCH_LIMIT = 200


class PolymarketDataIngestion:
    """Асинхронний клієнт для збору даних з REST API Polymarket."""

    def __init__(self, timeout: int = config.HTTP_TIMEOUT):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        ensure_data_dirs()

    @staticmethod
    def resolve_tag_slugs(category: Optional[str]) -> List[str]:
        """Map a UI category name to Polymarket API tag_slug list."""
        if not category or category == "All":
            return []
        return list(CATEGORY_TAG_MAP.get(category, []))

    @staticmethod
    def recommended_limit(category: Optional[str], requested: int) -> int:
        """Raise fetch limit for niche categories (e.g. Macro → up to 200)."""
        if category in NICHE_CATEGORIES:
            return max(int(requested), NICHE_FETCH_LIMIT)
        return int(requested)

    async def fetch_active_events(
        self,
        limit: int = 20,
        *,
        category: Optional[str] = None,
        tag_slugs: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Fetch active events. When ``category`` / ``tag_slugs`` is set, query the
        Gamma API with ``tag_slug`` (per tag) instead of post-filtering top-N.
        """
        slugs = tag_slugs if tag_slugs is not None else self.resolve_tag_slugs(category)
        fetch_limit = self.recommended_limit(category, limit)

        if not slugs:
            params = {
                "limit": fetch_limit,
                "active": "true",
                "closed": "false",
            }
            return await self._get_events(params)

        # One request per tag_slug, then dedupe by event id / slug
        batches = await asyncio.gather(
            *[
                self._get_events(
                    {
                        "limit": fetch_limit,
                        "active": "true",
                        "closed": "false",
                        "tag_slug": slug,
                    }
                )
                for slug in slugs
            ]
        )
        merged: Dict[str, Dict] = {}
        for batch in batches:
            for event in batch:
                key = str(event.get("id") or event.get("slug") or event.get("title") or id(event))
                if key not in merged:
                    merged[key] = event
        events = list(merged.values())
        logger.info(
            "Fetched %d unique active events for category=%s tags=%s (limit=%d)",
            len(events),
            category,
            slugs,
            fetch_limit,
        )
        return events[:fetch_limit]

    async def fetch_closed_events(self, limit: int = 50) -> List[Dict]:
        """Отримання закритих (resolved) подій для історичного бектесту."""
        params = {"limit": limit, "closed": "true", "active": "false"}
        return await self._get_events(params)

    async def _get_events(self, params: Dict) -> List[Dict]:
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(config.GAMMA_API_URL, params=params) as response:
                    if response.status == 200:
                        payload = await response.json()
                        return payload if isinstance(payload, list) else []
                    response.raise_for_status()
            except Exception as e:
                logger.error("Error fetching events: %s", e)
                return []
        return []

    async def fetch_order_book(self, token_id: str) -> Dict:
        """Отримання глибини стакана ордерів (Order Book) для конкретного токена."""
        params = {"token_id": token_id}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(config.CLOB_API_URL, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    return {"bids": [], "asks": []}
            except Exception as e:
                logger.error("Error fetching order book for %s: %s", token_id, e)
                return {"bids": [], "asks": []}

    def process_market_data(self, events: List[Dict], *, closed: bool = False) -> pd.DataFrame:
        """Перетворення сирого JSON у структурований DataFrame."""
        records = []
        for event in events:
            event_title = event.get("title", "")
            event_tags = self._extract_event_tags(event)
            event_category = self._category_from_tags(event_tags)
            for market in event.get("markets", []):
                question = market.get("question", "")

                clob_token_ids = market.get("clobTokenIds", [])
                if isinstance(clob_token_ids, str):
                    try:
                        clob_token_ids = json.loads(clob_token_ids)
                    except json.JSONDecodeError:
                        clob_token_ids = []

                outcome_prices = market.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except json.JSONDecodeError:
                        outcome_prices = []

                if len(clob_token_ids) < 2 or len(outcome_prices) < 2:
                    continue

                try:
                    price_yes = float(outcome_prices[0]) if outcome_prices[0] is not None else 0.5
                    price_no = float(outcome_prices[1]) if outcome_prices[1] is not None else 0.5
                    volume = float(market.get("volume", 0) or 0)
                    liquidity = float(market.get("liquidity", 0) or 0)
                except (ValueError, TypeError):
                    continue

                outcome = None
                if closed:
                    if price_yes >= 0.95:
                        outcome = 1
                    elif price_yes <= 0.05:
                        outcome = 0

                records.append(
                    {
                        "event_title": event_title,
                        "question": question,
                        "token_id_yes": clob_token_ids[0],
                        "token_id_no": clob_token_ids[1],
                        "price_yes": price_yes,
                        "price_no": price_no,
                        "volume": volume,
                        "liquidity": liquidity,
                        "end_date": market.get("endDate"),
                        "outcome": outcome,
                        "closed": closed,
                        "tags": ",".join(event_tags),
                        "category": event_category,
                    }
                )

        return pd.DataFrame(records)

    @staticmethod
    def _extract_event_tags(event: Dict) -> List[str]:
        tags: List[str] = []
        raw = event.get("tags") or []
        if isinstance(raw, list):
            for t in raw:
                if isinstance(t, dict):
                    slug = (t.get("slug") or t.get("label") or t.get("name") or "").strip().lower()
                    if slug:
                        tags.append(slug)
                elif isinstance(t, str) and t.strip():
                    tags.append(t.strip().lower())
        return tags

    @staticmethod
    def _category_from_tags(tags: List[str]) -> str:
        tag_set = set(tags)
        for category, slugs in CATEGORY_TAG_MAP.items():
            if category == "Other":
                continue
            if tag_set.intersection(slugs):
                return category
        return "Other"

    def save_snapshot(self, df: pd.DataFrame, prefix: str = "markets") -> str:
        """Persist a market snapshot as CSV under data/raw/."""
        ensure_data_dirs()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(config.RAW_DATA_DIR, f"{prefix}_{ts}.csv")
        df.to_csv(path, index=False)
        logger.info("Saved snapshot: %s (%d rows)", path, len(df))
        return path

    def load_latest_snapshot(self, prefix: str = "markets") -> pd.DataFrame:
        ensure_data_dirs()
        files = sorted(
            f
            for f in os.listdir(config.RAW_DATA_DIR)
            if f.startswith(prefix) and f.endswith(".csv")
        )
        if not files:
            return pd.DataFrame()
        path = os.path.join(config.RAW_DATA_DIR, files[-1])
        return pd.read_csv(path)

    def generate_synthetic_market_data(self, num_markets: int = 50, seed: int = None) -> pd.DataFrame:
        """
        Synthetic cross-section of binary contracts with spot/strike/tenor fields
        for Black-Scholes + Breeden-Litzenberger pipelines.
        """
        rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
        rows = []
        for i in range(num_markets):
            spot = float(rng.uniform(80, 120))
            strike = float(spot * rng.uniform(0.85, 1.15))
            tenor = float(rng.uniform(0.05, 0.75))
            # Market-implied YES ≈ digital price with noise
            moneyness = (spot - strike) / max(spot, 1e-6)
            base_p = float(1.0 / (1.0 + np.exp(-8.0 * moneyness)))
            price_yes = float(np.clip(base_p + rng.normal(0, 0.06), 0.05, 0.95))
            price_no = float(np.clip(1.0 - price_yes + rng.uniform(-0.03, 0.03), 0.05, 0.95))
            outcome = int(rng.random() < base_p)
            rows.append(
                {
                    "event_title": f"Synthetic Event {i + 1}",
                    "question": f"Will underlying finish above {strike:.1f}? (#{i + 1})",
                    "token_id_yes": f"syn_yes_{i}",
                    "token_id_no": f"syn_no_{i}",
                    "price_yes": round(price_yes, 4),
                    "price_no": round(price_no, 4),
                    "volume": float(rng.lognormal(8.0, 1.0)),
                    "liquidity": float(rng.lognormal(7.0, 1.0)),
                    "end_date": None,
                    "underlying_spot": round(spot, 4),
                    "strike_price": round(strike, 4),
                    "tenor_years": round(tenor, 4),
                    "outcome": outcome,
                    "category": ["Crypto", "Macro", "Politics", "Other"][i % 4],
                }
            )
        return pd.DataFrame(rows)

    def generate_historical_timeseries(
        self,
        num_contracts: int = 20,
        days: int = 60,
        seed: int = None,
    ) -> pd.DataFrame:
        """
        Synthetic daily time series of contract prices converging to outcomes.
        Used for chronological backtesting when live history is unavailable.
        """
        rng = np.random.default_rng(config.RANDOM_STATE if seed is None else seed)
        start = datetime.now(timezone.utc) - timedelta(days=days)
        frames = []
        for c in range(num_contracts):
            true_p = float(rng.uniform(0.2, 0.8))
            outcome = int(rng.random() < true_p)
            price = float(rng.uniform(0.25, 0.75))
            for d in range(days):
                # Mean-revert toward true probability, then pin near outcome at the end
                progress = d / max(days - 1, 1)
                target = (1.0 - progress) * true_p + progress * (0.97 if outcome else 0.03)
                price = float(np.clip(price + 0.15 * (target - price) + rng.normal(0, 0.02), 0.02, 0.98))
                ts = start + timedelta(days=d)
                frames.append(
                    {
                        "timestamp": ts.isoformat(),
                        "contract_id": f"C{c:03d}",
                        "question": f"Historical contract #{c + 1}",
                        "price_yes": round(price, 4),
                        "price_no": round(float(np.clip(1.0 - price + rng.uniform(-0.02, 0.02), 0.02, 0.98)), 4),
                        "volume": float(rng.lognormal(7.5, 0.8)),
                        "liquidity": float(rng.lognormal(6.5, 0.8)),
                        "outcome": outcome,
                        "day_index": d,
                        "category": ["Crypto", "Macro", "Politics"][c % 3],
                    }
                )
        df = pd.DataFrame(frames)
        path = os.path.join(config.PROCESSED_DATA_DIR, "historical_timeseries.csv")
        df.to_csv(path, index=False)
        logger.info("Wrote synthetic historical timeseries: %s (%d rows)", path, len(df))
        return df

    async def collect_and_persist(self, active_limit: int = 30, closed_limit: int = 40) -> Dict[str, str]:
        """Fetch live + closed markets from REST API and save CSV snapshots."""
        active_events, closed_events = await asyncio.gather(
            self.fetch_active_events(limit=active_limit),
            self.fetch_closed_events(limit=closed_limit),
        )
        active_df = self.process_market_data(active_events, closed=False)
        closed_df = self.process_market_data(closed_events, closed=True)
        paths = {}
        if not active_df.empty:
            paths["active"] = self.save_snapshot(active_df, prefix="active_markets")
        if not closed_df.empty:
            paths["closed"] = self.save_snapshot(closed_df, prefix="closed_markets")
        # Always refresh offline timeseries for reproducible backtests
        ts = self.generate_historical_timeseries()
        paths["timeseries"] = os.path.join(config.PROCESSED_DATA_DIR, "historical_timeseries.csv")
        paths["timeseries_rows"] = str(len(ts))
        return paths


if __name__ == "__main__":
    async def _main():
        ingester = PolymarketDataIngestion()
        print("Collecting Polymarket snapshots...")
        paths = await ingester.collect_and_persist()
        print(paths)

    asyncio.run(_main())
