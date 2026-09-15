"""
main.py - Main Pipeline Orchestrator
"""
import logging
from config import config
from src.data_ingestion import PolymarketDataIngestion
from src.math_engine import BreedenLitzenbergerEngine, BlackScholesBinaryEngine
from src.anomaly_detector import MispricingAnomalyDetector
from src.backtester import QuantitativeBacktester
from src.visualizer import QuantVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_quant_pipeline(num_markets: int = 50):
    logger.info("Initializing Quantitative Engine Pipeline...")
    
    # 1. Збір та генерація даних
    ingester = PolymarketDataIngestion()
    df = ingester.generate_synthetic_market_data(num_markets=num_markets)

    # 2. Кількісні розрахунки Black-Scholes
    bl_engine = BreedenLitzenbergerEngine()
    bs_engine = BlackScholesBinaryEngine()

    fair_prices, deltas = [], []
    for _, row in df.iterrows():
        S, K, T = row["underlying_spot"], row["strike_price"], row["tenor_years"]
        fair_prices.append(round(bs_engine.price_binary_call(S, K, T, config.RISK_FREE_RATE, config.DEFAULT_VOLATILITY), 4))
        deltas.append(bs_engine.compute_binary_greeks(S, K, T, config.RISK_FREE_RATE, config.DEFAULT_VOLATILITY)["delta"])

    df["fair_price"] = fair_prices
    df["delta"] = deltas

    # 3. Вилучення щільності ймовірностей (RND)
    grid_k, pdf_rnd, _ = bl_engine.extract_density_and_delta(df["strike_price"].values, df["fair_price"].values, tenor=config.DEFAULT_TENOR)

    # 4. ML Детекція аномалій
    detector = MispricingAnomalyDetector()
    df_analyzed = detector.fit_predict(df)

    # 5. Симуляція бектестингу
    backtester = QuantitativeBacktester()
    backtest_results = backtester.run_backtest(df_analyzed)

    # 6. Генерація графічного дашборду
    logger.info("Rendering visual analytics and backtest dashboard...")
    visualizer = QuantVisualizer(output_dir="reports/charts")
    
    path_bt = visualizer.plot_backtest_dashboard(backtest_results)
    path_rnd = visualizer.plot_breeden_litzenberger_rnd(grid_k, pdf_rnd)
    path_anom = visualizer.plot_anomaly_detection(df_analyzed)

    logger.info(f"Charts saved in: {path_bt}, {path_rnd}, {path_anom}")
    logger.info(f"Pipeline Completed | PnL: ${backtest_results['total_pnl']} | ROI: {backtest_results['roi_percent']}% | Sharpe: {backtest_results['sharpe_ratio']}")

if __name__ == "__main__":
    run_quant_pipeline(num_markets=50)