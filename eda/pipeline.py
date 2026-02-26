# EDA/pipeline.py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path

from data_io import (
    load_coinmetrics,
    load_parquet,
    integrity_report,
    save_csv,
    setup_output_dirs,
)

from analysis import (
    add_coinmetrics_features,
    build_polymarket_daily_panel,
    aggregate_polymarket_daily,
    plot_line,
    plot_hist,
    corr_matrix,
    corr_heatmap,
    correlation_summary,
    run_pca,
    run_baseline_models,
    run_strategy_evaluation,
)


def run_pipeline(data_root, eda_root, max_var_plots):
    data_root = Path(data_root)
    tables, plots = setup_output_dirs(eda_root)

    cm = load_coinmetrics(data_root / "Coin Metrics" / "coinmetrics_btc.csv")
    markets = load_parquet(data_root / "Polymarket" / "finance_politics_markets.parquet")
    odds = load_parquet(data_root / "Polymarket" / "finance_politics_odds_history.parquet")

    save_csv(integrity_report(cm, "coinmetrics"), tables / "coinmetrics_integrity.csv")
    if markets is not None:
        save_csv(integrity_report(markets, "polymarket_markets"), tables / "polymarket_markets_integrity.csv")
    if odds is not None:
        save_csv(integrity_report(odds, "polymarket_odds"), tables / "polymarket_odds_integrity.csv")

    cm = add_coinmetrics_features(cm)
    save_csv(cm, tables / "coinmetrics_features.csv")

    pm_panel = build_polymarket_daily_panel(markets, odds)
    save_csv(pm_panel, tables / "polymarket_daily_panel.csv")

    if "note" in pm_panel.columns:
        save_csv(pm_panel, tables / "polymarket_note.csv")
        return

    pm_daily = aggregate_polymarket_daily(pm_panel)
    save_csv(pm_daily, tables / "polymarket_daily_signals.csv")

    joined = cm.merge(pm_daily, on="date", how="left").sort_values("date")
    save_csv(joined, tables / "btc_polymarket_joined_daily.csv")

    joined_t = joined.copy()
    for c in joined_t.select_dtypes(include="number").columns:
        joined_t[c] = pd.to_numeric(joined_t[c], errors="coerce")

    if "PriceUSD" in joined_t.columns:
        px = pd.to_numeric(joined_t["PriceUSD"], errors="coerce").replace(0, np.nan)
        joined_t["log_price"] = np.log(px)

    diff_cols = [c for c in ["log_price", "TxCnt", "HashRate", "FeeTotUSD", "TxTfrValAdjUSD", "AdrActCnt"] if c in joined_t.columns]
    for c in diff_cols:
        joined_t[f"d_{c}"] = joined_t[c].diff()

    for c in ["pm_election_prob_vw", "pm_fed_prob_vw", "pm_crypto_prob_vw"]:
        if c in joined_t.columns:
            joined_t[f"d_{c}"] = pd.to_numeric(joined_t[c], errors="coerce").diff()

    save_csv(joined_t, tables / "joined_with_transforms.csv")

    plot_line(joined, "date", "PriceUSD", "Bitcoin price", plots / "btc_price.png")
    if "log_ret_1d" in joined.columns:
        plot_hist(joined["log_ret_1d"], "BTC daily log returns", plots / "btc_logret_hist.png")
    if "vol_30d" in joined.columns:
        plot_line(joined, "date", "vol_30d", "BTC 30 day rolling volatility", plots / "btc_vol_30d.png")

    core_cols = [c for c in [
        "log_ret_1d",
        "vol_30d",
        "pm_election_prob_chg_1d",
        "pm_fed_prob_chg_1d",
        "pm_crypto_prob_chg_1d",
        "TxCnt",
        "HashRate",
        "AdrActCnt",
    ] if c in joined.columns]

    if len(core_cols) >= 3:
        core_corr = corr_matrix(joined[core_cols])
        if core_corr is not None:
            core_corr.to_csv(tables / "corr_core.csv")
            corr_heatmap(core_corr, "Core correlations", plots / "corr_core.png")

    full_corr = corr_matrix(joined_t.select_dtypes(include="number"))
    if full_corr is not None:
        full_corr.to_csv(tables / "corr_transformed.csv")
        corr_heatmap(full_corr, "Correlations after transformations", plots / "corr_transformed.png")

    correlation_summary(joined_t, tables / "corr_summary.csv", target="log_ret_1d", top_n=30)

    run_pca(joined_t, tables, plots, prefix="pca", n_components=8, min_rows=250)
    run_baseline_models(joined_t, tables / "baseline_model_results.csv", min_rows=250)

    run_strategy_evaluation(joined, tables, plots)

    num_cols = joined.select_dtypes(include="number").columns.tolist()
    num_cols = num_cols[:max_var_plots]
    for c in num_cols:
        plot_line(joined, "date", c, f"time series {c}", plots / f"ts_{c}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--eda-root", default="EDA")
    parser.add_argument("--max-var-plots", type=int, default=50)
    args = parser.parse_args()

    run_pipeline(args.data_root, args.eda_root, args.max_var_plots)