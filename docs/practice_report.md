# Звіт / відповідність завданню практики

## 1. Інструктаж з охорони праці, техніки безпеки та внутрішнього розпорядку

**Статус:** виконується на підприємстві / в організації практики (не в репозиторії).

Чекліст для звіту:

- [ ] Пройдено вступний інструктаж з ОП та ТБ (дата, ПІБ інструктора, підпис)
- [ ] Ознайомлено з правилами внутрішнього розпорядку
- [ ] Додано копію / витяг журналу інструктажів до паперового звіту

Робота з ПЗ: дотримуватись ергономіки робочого місця, режиму праці/відпочинку,
не зберігати секрети API в git.

## 2. Аналіз застосування методів обчислювального інтелекту для mispricing

Ринки передбачень котирують бінарні контракти з ціною ≈ суб’єктивною ймовірністю.
**Некоректне ціноутворення** проявляється як стійке відхилення ринкової ціни від
модельної справедливої ймовірності з урахуванням ліквідності та спреду.

### Математичний підхід (Breeden-Litzenberger)

З другої похідної ціни опціону за страйком отримують **ризик-нейтральну щільність**
$f(K)$. Для бінарних / digital контрактів справедлива ймовірність події оцінюється
через інтеграл RND / дисконтовану N(d2) у Black-Scholes.

### Методи обчислювального інтелекту

| Метод | Роль у проєкті |
|-------|----------------|
| Isolation Forest | Ненаглядове виявлення аномалій у просторі (Δціни, спред, volume, liquidity) |
| Feature engineering | `price_delta`, log-volume, divergence×liquidity |
| Імовірнісні метрики | Brier Score, LogLoss для порівняння прогнозів |
| Hybrid ML | Збільшення ваги fair_price при високому anomaly score |

Висновок: класична модель задає **інтерпретовану** fair value; ML додає
**адаптивне** виявлення структурних відхилень у мікроструктурі ринку.

## 3. Реалізований інтелектуальний модуль (Python)

- Справедлива ціна: `src/math_engine.py` (`BreedenLitzenbergerEngine`, `BlackScholesBinaryEngine`)
- Аномалії: `src/anomaly_detector.py` (`MispricingAnomalyDetector`)
- UI: `dashboard.py` + `src/visualization/charts.py`

## 4. Автоматичний збір та обробка даних (REST API)

- Gamma API: активні / закриті події  
- CLOB API: order book  
- Збереження: `data/raw/*.csv`, часові ряди `data/processed/historical_timeseries.csv`  
- Оркестрація: `PolymarketDataIngestion.collect_and_persist()` / `python main.py`

## 5. Порівняння точності BL та ML

Модуль `src/comparison.py` рахує Brier / LogLoss / MAE для:

1. market  
2. breeden_litzenberger  
3. ml_hybrid  

Результати: `reports/metrics/bl_vs_ml_comparison.csv` (+ `.json` з висновком).

## 6. Бектестинг на часових рядах

- Cross-section: `QuantitativeBacktester`  
- Chronological panel: `TimeSeriesBacktester.run_timeseries_backtest`  
- Метрики: Total PnL, ROI, Sharpe, Max Drawdown %, Brier, Win Rate  
- Графіки: `reports/charts/`

## 7. Процес розробки (Kanban + Git + GitHub)

Див. `docs/kanban.md`. Репозиторій на GitHub використовується як система контролю версій
із комітами за функціональними інкрементами.
