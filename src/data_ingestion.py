import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Optional


class PolymarketDataIngestion:
    """Асинхронний клієнт для збору даних з REST API Polymarket."""

    GAMMA_API_URL = "https://gamma-api.polymarket.com/events"
    CLOB_API_URL = "https://clob.polymarket.com/book"

    def __init__(self, timeout: int = 10):
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def fetch_active_events(self, limit: int = 20) -> List[Dict]:
        """Отримання списку активних подій та пов'язаних ринків."""
        params = {
            "limit": limit,
            "active": "true",
            "closed": "false"
        }
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.GAMMA_API_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                response.raise_for_status()

    async def fetch_order_book(self, token_id: str) -> Dict:
        """Отримання глибини стакана ордерів (Order Book) для конкретного токена."""
        params = {"token_id": token_id}
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            async with session.get(self.CLOB_API_URL, params=params) as response:
                if response.status == 200:
                    return await response.json()
                return {"bids": [], "asks": []}

    def process_market_data(self, events: List[Dict]) -> pd.DataFrame:
        """Перетворення сирого JSON у структурований DataFrame для подальшого аналізу."""
        records = []
        for event in events:
            event_title = event.get("title", "")
            for market in event.get("markets", []):
                question = market.get("question", "")
                clob_token_ids = market.get("clobTokenIds", [])
                outcome_prices = market.get("outcomePrices", [])

                if len(clob_token_ids) >= 2 and len(outcome_prices) >= 2:
                    records.append({
                        "event_title": event_title,
                        "question": question,
                        "token_id_yes": clob_token_ids[0],
                        "token_id_no": clob_token_ids[1],
                        "price_yes": float(outcome_prices[0]),
                        "price_no": float(outcome_prices[1]),
                        "volume": float(market.get("volume", 0)),
                        "liquidity": float(market.get("liquidity", 0)),
                        "end_date": market.get("endDate")
                    })
        return pd.DataFrame(records)


# Тестовий запуск модуля
if __name__ == "__main__":
    async def main():
        ingester = PolymarketDataIngestion()
        print("Завантаження активних ринків з Polymarket...")
        events = await ingester.fetch_active_events(limit=10)
        df = ingester.process_market_data(events)
        print(f"Завантажено {len(df)} активних контрактів.")
        print(df[["question", "price_yes", "price_no", "volume"]].head())

    asyncio.run(main())