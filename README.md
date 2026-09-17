# prediction_market_engine

An asynchronous quantitative analytics engine for identifying probability mispricings and arbitrage opportunities in prediction markets (Polymarket) using the Breeden-Litzenberger model and machine learning.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![MIT License](https://img.shields.io/badge/License-MIT-green)

## Overview

prediction_market_engine is designed to analyze prediction markets for probability mispricings and arbitrage opportunities. Leveraging both the Breeden-Litzenberger model and advanced machine learning techniques, it offers asynchronous data processing, anomaly detection, backtesting, and rich visualizations via an interactive dashboard. The engine integrates with Polymarket, enabling robust quantitative analytics and streamlined reporting.

## Tech Stack

- **Language**: Python
- **Core Libraries**:
  - aiohttp
  - requests
  - pandas
  - numpy
  - scipy
  - scikit-learn
  - xgboost
  - plotly
  - streamlit
  - matplotlib
  - pydantic
  - fastapi
  - uvicorn

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/c0n666/prediction_market_engine.git
   cd prediction_market_engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare raw data**
   - Place your raw market data files in the `data/raw/` directory.

2. **Run data ingestion and processing**
   ```bash
   python src/data_ingestion.py
   ```

3. **Start the analytics engine**
   - Using `main.py`:
     ```bash
     python main.py
     ```
   - Or using `app.py`:
     ```bash
     python app.py
     ```

4. **Access the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

5. **Review generated reports**
   - Output charts and metrics will be available in the `reports/` directory.

## Project Structure

```
prediction_market_engine/
├── .gitignore
├── LICENSE
├── app.py                # Application entry point
├── config.py             # Configuration settings
├── dashboard.py          # Streamlit dashboard interface
├── data/
│   ├── processed/
│   │   ├── .gitkeep
│   │   ├── analyzed_markets.csv
│   │   └── historical_timeseries.csv
│   └── raw/
│       └── .gitkeep
├── main.py               # Main analytics engine
├── reports/
│   ├── charts/
│   │   └── .gitkeep
│   └── metrics/
│       ├── .gitkeep
│       ├── bl_vs_ml_comparison.csv
│       └── bl_vs_ml_comparison.json
├── requirements.txt      # Python dependencies
├── src/
│   ├── __init__.py
│   ├── anomaly_detector.py
│   ├── backtester.py
│   ├── comparison.py
│   ├── data_ingestion.py
│   ├── math_engine.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── charts.py
│   └── visualizer.py
```

## API Endpoints

| Method | Endpoint                                         | Description                                         |
|--------|--------------------------------------------------|-----------------------------------------------------|
| GET    | `/`                                              | Main entry point for the application                |
| GET    | `/dashboard`                                     | Serves the interactive dashboard                    |
| GET    | `/data/processed/analyzed_markets.csv`           | Fetch analyzed market data                          |
| GET    | `/data/processed/historical_timeseries.csv`      | Fetch historical time series data                   |
| POST   | `/anomaly_detector`                              | Submit data for anomaly detection                   |
| POST   | `/backtester`                                    | Run backtesting on submitted data                   |
| POST   | `/comparison`                                    | Compare Breeden-Litzenberger model vs ML outputs    |
| GET    | `/reports/metrics/bl_vs_ml_comparison.csv`       | Get comparison metrics report (CSV)                 |
| GET    | `/reports/metrics/bl_vs_ml_comparison.json`      | Get comparison metrics report (JSON)                |

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with descriptive messages.
4. Push to your branch and open a Pull Request.

## License

MIT License. See [`LICENSE`](LICENSE) for details.

---
[![README powered by ReadmeAI](https://img.shields.io/badge/README-powered%20by%20ReadmeAI-4c9be8?style=flat-square&logo=markdown)](https://www.readmeai.in)
