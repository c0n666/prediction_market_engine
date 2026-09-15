# Prediction Market Mispricing Engine

Асинхронний quantitative analytics engine для виявлення некоректного ціноутворення
на ринках передбачень (Polymarket): модель **Breeden-Litzenberger** (ризик-нейтральні
щільності), **Black-Scholes** для бінарних контрактів, **Isolation Forest** (ML) та
бектестинг на часових рядах.

Репозиторій: [github.com/c0n666/prediction_market_engine](https://github.com/c0n666/prediction_market_engine)

## Відповідність завданню практики

| Вимога | Реалізація |
|--------|------------|
| Аналіз методів обчислювального інтелекту для mispricing | `docs/practice_report.md` |
| Модуль справедливої ціни + аномалій (Python) | `src/math_engine.py`, `src/anomaly_detector.py` |
| Автозбір через відкритий REST API | `src/data_ingestion.py` → Gamma / CLOB Polymarket |
| Історичні дані / часові ряди | `data/processed/historical_timeseries.csv`, snapshots у `data/raw/` |
| Порівняння BL vs ML | `src/comparison.py` → `reports/metrics/` |
| Бектест на часових рядах | `TimeSeriesBacktester` у `src/backtester.py` |
| Kanban + GIT + GitHub | `docs/kanban.md`, гілка `main` на GitHub |

> Інструктаж з охорони праці / ТБ / внутрішнього розпорядку — організаційний пункт
> практики (див. чекліст у `docs/practice_report.md`), не частина коду.

## Швидкий старт

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# Повний пайплайн (API + synthetic fallback + порівняння + бектест + графіки)
python main.py

# Інтерактивний дашборд
streamlit run dashboard.py

# Лише збір REST-знімків
python -m src.data_ingestion
```

## Архітектура

```
config.py
main.py                 # оркестратор
dashboard.py            # Streamlit UI
src/
  data_ingestion.py     # REST + historical/synthetic
  math_engine.py        # Breeden-Litzenberger + Black-Scholes binary
  anomaly_detector.py   # Isolation Forest
  comparison.py         # Brier / LogLoss / MAE: market vs BL vs ML
  backtester.py         # Strategy / Quantitative / TimeSeries
  visualization/        # Plotly (RND, equity, Greeks)
  visualizer.py         # Matplotlib report charts
data/raw|processed/
reports/charts|metrics/
docs/
```

## Метрики порівняння моделей

Після `python main.py` дивіться `reports/metrics/bl_vs_ml_comparison.csv`:

- **market** — сира ймовірність YES з ринку  
- **breeden_litzenberger** — fair P(YES) з RND / BS  
- **ml_hybrid** — зважене поєднання market + fair за anomaly score  

Нижчий **Brier Score** = краща калібровка ймовірностей.

## Ліцензія / навчальне використання

Навчальний проєкт (університетська практика). Не є інвестиційною порадою.
