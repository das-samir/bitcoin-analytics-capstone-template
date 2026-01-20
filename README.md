# Bitcoin Analytics Capstone: Polymarket Integration

This capstone challenges you to build a predictive Bitcoin DCA (Dollar Cost Averaging) strategy by integrating **Polymarket prediction data** with traditional on-chain metrics.

---

## Project (Trilemma Foundation): Improving Institutional Bitcoin Accumulation Strategies

*   **Problem:** Institutions are accumulating Bitcoin; standard DCA may be suboptimal under competitive, high-volume conditions.
*   **Goal:** Design **data-driven, long-only** accumulation strategies that keep DCA’s systematic discipline but **improve acquisition efficiency** within a fixed **budget + time horizon** (esp. institutional scale, e.g., $1M+/month).
*   **Core workstreams:** BTC primer → EDA of BTC properties → feature-informed **daily purchase schedules** → **backtesting + tuning** (optional: slippage/execution risk) → evaluation tooling + visual benchmarks vs DCA → open-source dashboards/resources.
*   **Data provided:** Market (price/OHLC/volume/BVOL), on-chain (UTXO/exchange flows/etc.), macro indicators, sentiment metrics; external data welcome after core exploration.
*   **Tech expectations:** Python-first; time-series/statistical analysis + ML; API/data pipelines; interactive dashboards (Dash/Plotly or Streamlit).
*   **Deliverables:** MIT-licensed open-source repo(s) with reproducible models, interactive dashboards, tutorials/notebooks/READMEs, and a final presentation.
*   **IP/Licensing:** Everything produced is open-sourced under **MIT** (contributors retain attribution).
*   **Contacts:** Mohammad Ashkani (Comms Lead) and Mateusz Faltyn (Technical Lead).


---
## The Capstone Objective

The goal of this project is to evolve a basic MVRV-based DCA model into a sophisticated, market-aware strategy.

**The Challenge:** Improve the integration of Polymarket data into the model to produce superior predictive signals.

### Your Tasks:
1. **Fork the Template**: Create your own version of this repository to build your model.
2. **Integrate Polymarket Data**: Leverage the provided Polymarket datasets (Politics, Finance, Crypto) to extract predictive signals.
3. **Direct Integration**: Modify the core model logic to incorporate these signals alongside traditional indicators like MVRV and Moving Averages.
4. **Outperform the Baseline**: Demonstrate through backtesting that your integrated model provides superior risk-adjusted outcomes compared to the foundation model.

---

## Getting Started

### 1. Repository Setup
1. **Fork this repository** to your own GitHub account.
2. Clone your fork locally:
   ```bash
   git clone <your-fork-url>
   cd bitcoin-analytics-capstone-template
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### 2. Data Acquisition
All necessary data is organized in the `data/` directory. You can download it automatically:
```bash
python data/download_data.py
```

**What's included:**
- **CoinMetrics BTC Data**: Historical price and MVRV metrics (`data/Coin Metrics/`).
- **Polymarket Data**: 5 Parquet files containing trades, odds history, and market metadata (`data/Polymarket/`).
- **Data Schemas**: Detailed documentation in `data/Polymarket/polymarket_btc_analytics_schema.md`.

---

## The Foundation (Baseline Model)

The repository provides a working **Foundation Model** located in `template/`. This model uses:
- **MVRV Z-score**: Buy more when undervalued (Deep Value Zone).
- **200-day MA**: Trend-following signal modulation.
- **Momentum/Volatility**: Acceleration and dampening modifiers.

### Running the Baseline
Before adding Polymarket info, run the current backtest to establish your baseline:
```bash
python -m template.backtest_template
```

---

## The Challenge (Polymarket Integration)

Your task is to modify `template/model_development_template.py` to integrate Polymarket signals. 

> **Important:** While the primary goal is Polymarket integration, you are also encouraged to make changes to the base model logic (MVRV, MA, etc.) if it helps you better incorporate the prediction market data and improve the model.

### Potential Signal Leads:
- **Election Probabilities**: How do presidential odds correlate with BTC volatility?
- **Economic Indicators**: Do prediction markets for Fed rate cuts lead BTC price movements?
- **Crypto Sentiment**: Use specific "Polymarket Crypto" markets as lead indicators for retail sentiment.

### Repository Workflow
```
.
├── template/                        # DIRECTORY TO FORK
│   ├── model_development_template.py # INTEGRATE POLYMARKET SIGNALS HERE
│   ├── backtest_template.py         # Evaluate your new strategy
│   ├── model_template.md            # Detailed MVRV logic documentation
│   └── backtest_template.md         # Backtest engine documentation
├── example_1/                       # REFERENCE IMPLEMENTATION
│   ├── model_development_example_1.py# Example Polymarket integration
│   ├── backtest_example_1.py        # Example backtest
│   ├── model_example_1.md           # Documentation for updated logic
│   └── backtest_example_1.md        # Documentation for Example 1 results
├── data/                            # Bitcoin & Polymarket source data
├── output/                          # Your strategy's performance visualizations
└── tests/                           # Ensure your model remains stable
```

---

## Example Implementation: `example_1`

To help you get started, we've provided `example_1/`. This is a complete "fork" of the `template/` directory that demonstrates:
1. **Data Loading**: How to use `prelude_example_1.py` to ingest Polymarket parquet files.
2. **Signal Generation**: A concrete example of mapping Polymarket odds to model modifiers.
3. **Backtesting**: Running the evaluation suite on the integrated model.

**Study `example_1/` to understand the workflow before building your own model in a new folder.**

---

## Evaluation Metrics

Your integrated model will be evaluated on the following (automated via `backtest_template.py`):
- **Win Rate**: Must outperform uniform DCA in >50% of 1-year windows.
- **SPD Percentile**: Overall efficiency of satoshi accumulation.
- **Model Score**: A combination of win rate and reward-to-risk percentile.

---

## Documentation
- **Model Logic**: See `docs/model.md` for the current MVRV implementation.
- **Backtest Framework**: See `docs/model_backtest.md` for scoring methodology.
- **Polymarket Schema**: See `data/Polymarket/polymarket_btc_analytics_schema.md`.

---
*Developed for the Bitcoin Analytics Capstone.*
