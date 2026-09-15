import aiohttp
import asyncio
import json
import logging
import pandas as pd
from typing import Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)


class PolymarketDataIngestion:
    """Асинхронний клієнт для збору даних з REST API Polymarket."""

    def __init__(self, timeout: int = config.HTTP_TIMEOUT):
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_active_events(self, limit: int = 20) -> List[Dict]:
        """Отримання списку активних подій та пов'язаних ринків."""
        params = {
            "limit": limit,
            "active": "true",
            "closed": "false"
        }
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(config.GAMMA_API_URL, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    response.raise_for_status()
            except Exception as e:
                logger.error(f"Error fetching active events: {e}")
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
                logger.error(f"Error fetching order book for {token_id}: {e}")
                return {"bids": [], "asks": []}

    def process_market_data(self, events: List[Dict]) -> pd.DataFrame:
        """Перетворення сирого JSON у структурований DataFrame з розпаршуванням рядкових полів."""
        records = []
        for event in events:
            event_title = event.get("title", "")
            for market in event.get("markets", []):
                question = market.get("question", "")
                
                # Парсинг clobTokenIds (може бути як списком, так і закодованим рядком)
                clob_token_ids = market.get("clobTokenIds", [])
                if isinstance(clob_token_ids, str):
                    try:
                        clob_token_ids = json.loads(clob_token_ids)
                    except json.JSONDecodeError:
                        clob_token_ids = []

                # Парсинг outcomePrices (часто повертається як JSON-рядок)
                outcome_prices = market.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except json.JSONDecodeError:
                        outcome_prices = []

                if len(clob_token_ids) >= 2 and len(outcome_prices) >= 2:
                    try:
                        price_yes = float(outcome_prices[0]) if outcome_prices[0] is not None else 0.5
                        price_no = float(outcome_prices[1]) if outcome_prices[1] is not None else 0.5
                        volume = float(market.get("volume", 0) or 0)
                        liquidity = float(market.get("liquidity", 0) or 0)
                    except (ValueError, TypeError):
                        continue

                    records.append({
                        "event_title": event_title,
                        "question": question,
                        "token_id_yes": clob_token_ids[0],
                        "token_id_no": clob_token_ids[1],
                        "price_yes": price_yes,
                        "price_no": price_no,
                        "volume": volume,
                        "liquidity": liquidity,
                        "end_date": market.get("endDate")
                    })

        return pd.DataFrame(records)


if __name__ == "__main__":
    async def main():
        ingester = PolymarketDataIngestion()
        print("Завантаження активних ринків з Polymarket...")
        events = await ingester.fetch_active_events(limit=10)
        df = ingester.process_market_data(events)
        print(f"Успішно завантажено {len(df)} активних контрактів.")
        if not df.empty:
            print(df[["question", "price_yes", "price_no", "volume"]].head())

    asyncio.run(main())