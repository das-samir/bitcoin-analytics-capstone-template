# EDA/analysis.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from strategy_eval import CapitalAssumptions, evaluate_strategies


def add_coinmetrics_features(cm: pd.DataFrame) -> pd.DataFrame:
    cm = cm.copy()
    if "date" in cm.columns:
        cm["date"] = pd.to_datetime(cm["date"], errors="coerce").dt.tz_localize(None)
    if "PriceUSD" in cm.columns:
        px = pd.to_numeric(cm["PriceUSD"], errors="coerce").replace(0, np.nan)
        cm["log_ret_1d"] = np.log(px).diff()
        cm["vol_30d"] = cm["log_ret_1d"].rolling(30).std()
    return cm


def _norm_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.lower()


def _to_datetime_utc_from_numeric(ts: pd.Series) -> pd.Series:
    x = pd.to_numeric(ts, errors="coerce")
    if x.notna().sum() == 0:
        return pd.to_datetime(ts, errors="coerce", utc=True)

    med = float(x.dropna().median())
    if med > 1e14:
        unit = "us"
    elif med > 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(x, unit=unit, utc=True)


def build_polymarket_daily_panel(markets: Optional[pd.DataFrame], odds: Optional[pd.DataFrame]) -> pd.DataFrame:
    if markets is None or odds is None:
        return pd.DataFrame({"note": ["missing polymarket inputs"]})

    m = markets.copy()
    o = odds.copy()

    for c in ["market_id", "token_id"]:
        if c in m.columns:
            m[c] = m[c].astype(str)
        if c in o.columns:
            o[c] = o[c].astype(str)

    if "timestamp" in o.columns:
        if np.issubdtype(o["timestamp"].dtype, np.datetime64):
            o["timestamp"] = pd.to_datetime(o["timestamp"], utc=True, errors="coerce")
        else:
            o["timestamp"] = _to_datetime_utc_from_numeric(o["timestamp"])

    o = o.dropna(subset=["timestamp", "price"])
    o["date"] = o["timestamp"].dt.floor("D").dt.tz_convert(None)
    o["price"] = pd.to_numeric(o["price"], errors="coerce")

    join_cols = [c for c in ["market_id", "question", "slug", "event_slug", "category", "volume", "active", "closed"] if c in m.columns]
    mm = m[join_cols].copy()
    if "volume" in mm.columns:
        mm["volume"] = pd.to_numeric(mm["volume"], errors="coerce")

    j = o.merge(mm, on="market_id", how="left")

    if len(j) == 0:
        return pd.DataFrame({"note": ["no polymarket odds after cleaning"]})

    q = _norm_text(j.get("question", pd.Series("", index=j.index)))
    cat = _norm_text(j.get("category", pd.Series("", index=j.index)))
    slug = _norm_text(j.get("slug", pd.Series("", index=j.index)))
    event = _norm_text(j.get("event_slug", pd.Series("", index=j.index)))

    j["is_election"] = (
        cat.str.contains("politic")
        | q.str.contains("election|primary|president|trump|biden|harris|gop|democrat|republican|senate|house")
        | event.str.contains("election|president")
    )

    j["is_fed_rates"] = (
        q.str.contains("fed|fomc|rate cut|rate hike|interest rate|cpi|inflation")
        | event.str.contains("fed|rates|fomc|inflation|cpi")
        | slug.str.contains("fed|rates|fomc")
    )

    j["is_crypto_sentiment"] = (
        cat.str.contains("crypto")
        | q.str.contains("bitcoin|btc|crypto|ethereum|eth|solana|memecoin|altcoin")
        | slug.str.contains("bitcoin|btc|crypto|eth")
    )

    j["w"] = 1.0
    if "volume" in j.columns:
        j["w"] = pd.to_numeric(j["volume"], errors="coerce").fillna(1.0).clip(lower=0.0)

    return j


def aggregate_polymarket_daily(pm_panel: pd.DataFrame) -> pd.DataFrame:
    if "note" in pm_panel.columns:
        return pm_panel

    df = pm_panel.copy()

    def _agg(mask_col: str, prefix: str) -> pd.DataFrame:
        sub = df[df[mask_col]].copy()
        if len(sub) == 0:
            return pd.DataFrame({"date": sorted(df["date"].unique())})

        g = sub.groupby("date", as_index=False).apply(
            lambda x: pd.Series(
                {
                    f"{prefix}_prob_vw": np.average(x["price"], weights=x["w"]) if x["price"].notna().any() else np.nan,
                    f"{prefix}_prob_mean": float(np.nanmean(x["price"])),
                    f"{prefix}_prob_std": float(np.nanstd(x["price"])),
                    f"{prefix}_n_markets": int(x["market_id"].nunique()),
                }
            )
        )
        g = g.reset_index(drop=True)
        g = g.sort_values("date")
        g[f"{prefix}_prob_chg_1d"] = g[f"{prefix}_prob_vw"].diff()
        return g

    all_dates = pd.DataFrame({"date": sorted(df["date"].unique())})

    election = _agg("is_election", "pm_election")
    fed = _agg("is_fed_rates", "pm_fed")
    crypto = _agg("is_crypto_sentiment", "pm_crypto")

    out = all_dates.merge(election, on="date", how="left").merge(fed, on="date", how="left").merge(crypto, on="date", how="left")
    return out


def plot_line(df: pd.DataFrame, x: str, y: str, title: str, outpath: Path) -> None:
    if y not in df.columns or x not in df.columns:
        return
    s = df[[x, y]].dropna()
    if len(s) < 5:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(pd.to_datetime(s[x]), pd.to_numeric(s[y], errors="coerce"))
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_hist(series: pd.Series, title: str, outpath: Path) -> None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 20:
        return
    plt.figure(figsize=(8, 4))
    plt.hist(s, bins=50)
    plt.title(title)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def corr_matrix(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    return num.corr()


def corr_heatmap(corr: pd.DataFrame, title: str, outpath: Path, vmax: float = 1.0) -> None:
    if corr is None or corr.shape[0] < 2:
        return
    plt.figure(figsize=(10, 8))
    plt.imshow(corr.values, aspect="auto", vmin=-vmax, vmax=vmax)
    plt.title(title)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=7)
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


def correlation_summary(df: pd.DataFrame, out_csv: Path, target: str = "log_ret_1d", top_n: int = 25) -> None:
    if target not in df.columns:
        return
    num = df.select_dtypes(include="number").copy()
    if target not in num.columns:
        return
    c = num.corr()[target].dropna().sort_values(key=lambda x: x.abs(), ascending=False)
    c = c.drop(index=target, errors="ignore").head(top_n)
    out = c.reset_index()
    out.columns = ["feature", "corr_with_target"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)


def run_pca(df: pd.DataFrame, tables_dir: Path, plots_dir: Path, prefix: str = "pca", n_components: int = 8, min_rows: int = 250) -> None:
    num = df.select_dtypes(include="number").copy()
    num = num.dropna(axis=1, how="all")
    num = num.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(num) < min_rows or num.shape[1] < 3:
        (tables_dir / f"{prefix}_note.csv").parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"note": [f"not enough rows or columns for PCA. rows={len(num)} cols={num.shape[1]}"]}).to_csv(
            tables_dir / f"{prefix}_note.csv",
            index=False,
        )
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(num.values)

    pca = PCA(n_components=min(n_components, X.shape[1]))
    Z = pca.fit_transform(X)

    evr = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cum_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
        }
    )
    evr.to_csv(tables_dir / f"{prefix}_explained_variance.csv", index=False)

    loadings = pd.DataFrame(pca.components_.T, index=num.columns, columns=evr["component"])
    loadings.to_csv(tables_dir / f"{prefix}_loadings.csv", index=True)

    plt.figure(figsize=(10, 4))
    plt.plot(evr["component"], evr["cum_explained_variance_ratio"], marker="o")
    plt.title("PCA cumulative explained variance")
    plt.xlabel("component")
    plt.ylabel("cumulative explained variance ratio")
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plots_dir / f"{prefix}_cum_explained_variance.png", dpi=150)
    plt.close()

    if Z.shape[1] >= 2:
        plt.figure(figsize=(6, 5))
        plt.scatter(Z[:, 0], Z[:, 1], s=8)
        plt.title("PCA scores, PC1 vs PC2")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(plots_dir / f"{prefix}_pc1_pc2.png", dpi=150)
        plt.close()


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def run_baseline_models(df: pd.DataFrame, out_csv: Path, min_rows: int = 200) -> None:
    if "log_ret_1d" not in df.columns:
        pd.DataFrame({"note": ["missing target log_ret_1d"]}).to_csv(out_csv, index=False)
        return

    num = df.select_dtypes(include="number").copy()
    num = num.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if len(num) < min_rows or num.shape[1] < 4:
        pd.DataFrame({"note": [f"not enough rows for models. rows={len(num)} cols={num.shape[1]}"]}).to_csv(out_csv, index=False)
        return

    y = num["log_ret_1d"].values
    X = num.drop(columns=["log_ret_1d"]).values

    split = int(0.8 * len(num))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    models = {
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.0005, max_iter=50_000),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=7, n_jobs=-1, max_depth=8),
    }

    rows = []
    for name, mdl in models.items():
        mdl.fit(X_tr, y_tr)
        pred = mdl.predict(X_te)
        rows.append(
            {
                "model": name,
                "rmse": _rmse(y_te, pred),
                "mae": float(mean_absolute_error(y_te, pred)),
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_features": int(X_tr.shape[1]),
            }
        )

    pd.DataFrame(rows).sort_values(["rmse", "mae"]).to_csv(out_csv, index=False)


def run_strategy_evaluation(
    joined: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    assumptions: Optional[CapitalAssumptions] = None,
) -> None:
    assumptions = assumptions or CapitalAssumptions(start_cash_usd=0.0, daily_budget_usd=10.0, fee_bps=10.0, slippage_bps=5.0)

    signal_cols = {
        "Election signal tilt": "pm_election_prob_chg_1d",
        "Fed rates signal tilt": "pm_fed_prob_chg_1d",
        "Crypto sentiment tilt": "pm_crypto_prob_chg_1d",
        "Volatility tilt": "vol_30d",
    }

    summary, wins_long, paths = evaluate_strategies(
        joined=joined,
        assumptions=assumptions,
        signal_cols=signal_cols,
        date_col="date",
        price_col="PriceUSD",
    )

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(tables_dir / "strategy_summary.csv", index=False)
    wins_long.to_csv(tables_dir / "strategy_win_windows.csv", index=False)

    for name, path in paths.items():
        safe = name.lower().replace(" ", "_").replace("/", "_")
        path.to_csv(tables_dir / f"strategy_path_{safe}.csv", index=False)

    if len(summary):
        plt.figure(figsize=(8, 4))
        plt.bar(summary["strategy"], summary["spd_sats_per_usd"])
        plt.title("SPD, sats per USD")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "strategy_spd.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(summary["strategy"], summary["win_rate_1y_windows"])
        plt.title("Win rate vs DCA across 1 year windows")
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plots_dir / "strategy_win_rate.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(summary["strategy"], summary["model_score"])
        plt.title("Model Score, win rate plus risk adjusted return")
        plt.xticks(rotation=25, ha="right")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(plots_dir / "strategy_model_score.png", dpi=150)
        plt.close()

    for name, path in paths.items():
        if "portfolio_usd" not in path.columns:
            continue
        plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(path["date"]), path["portfolio_usd"])
        plt.title(f"Equity curve, {name}")
        plt.xlabel("date")
        plt.ylabel("portfolio_usd")
        plt.tight_layout()
        safe = name.lower().replace(" ", "_").replace("/", "_")
        plt.savefig(plots_dir / f"equity_{safe}.png", dpi=150)
        plt.close()