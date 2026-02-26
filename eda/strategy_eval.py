# EDA/strategy_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


SATOSHIS_PER_BTC = 100_000_000


@dataclass(frozen=True)
class CapitalAssumptions:
    start_cash_usd: float = 0.0
    daily_budget_usd: float = 10.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    @property
    def cost_bps(self) -> float:
        return float(self.fee_bps + self.slippage_bps)

    @property
    def cost_mult(self) -> float:
        return 1.0 + (self.cost_bps / 10_000.0)


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _safe_price(px: pd.Series) -> pd.Series:
    px = pd.to_numeric(px, errors="coerce")
    return px.where(px > 0)


def _signal_to_multiplier(
    signal: pd.Series,
    low_q: float = 0.25,
    high_q: float = 0.75,
    low_mult: float = 0.5,
    mid_mult: float = 1.0,
    high_mult: float = 1.5,
) -> pd.Series:
    s = pd.to_numeric(signal, errors="coerce")
    q_low = s.quantile(low_q)
    q_high = s.quantile(high_q)
    mult = pd.Series(np.nan, index=s.index, dtype="float64")
    mult[s <= q_low] = low_mult
    mult[(s > q_low) & (s < q_high)] = mid_mult
    mult[s >= q_high] = high_mult
    return mult.fillna(mid_mult)


def simulate_accumulation(
    df: pd.DataFrame,
    assumptions: CapitalAssumptions,
    budget_mult: pd.Series,
    price_col: str = "PriceUSD",
    date_col: str = "date",
) -> pd.DataFrame:
    _require_cols(df, [date_col, price_col])

    out = df[[date_col, price_col]].copy()
    out[price_col] = _safe_price(out[price_col])
    out = out.dropna(subset=[price_col]).sort_values(date_col).reset_index(drop=True)

    mult = pd.to_numeric(budget_mult.reindex(out.index), errors="coerce").fillna(1.0).clip(0.0, 10.0)

    daily_usd = assumptions.daily_budget_usd * mult
    cost_mult = assumptions.cost_mult
    usd_spent = daily_usd * cost_mult
    btc_bought = daily_usd / out[price_col]
    sats_bought = btc_bought * SATOSHIS_PER_BTC

    out["budget_mult"] = mult.values
    out["daily_budget_usd"] = daily_usd.values
    out["usd_spent"] = usd_spent.values
    out["btc_bought"] = btc_bought.values
    out["sats_bought"] = sats_bought.values

    out["cum_usd_spent"] = out["usd_spent"].cumsum() + float(assumptions.start_cash_usd)
    out["cum_sats"] = out["sats_bought"].cumsum()
    out["cum_btc"] = out["cum_sats"] / SATOSHIS_PER_BTC
    out["portfolio_usd"] = out["cum_btc"] * out[price_col]

    out["spd"] = out["cum_sats"] / out["cum_usd_spent"].replace(0, np.nan)

    return out


def rolling_win_rate(
    strat: pd.DataFrame,
    dca: pd.DataFrame,
    window_days: int = 365,
    date_col: str = "date",
    value_col: str = "portfolio_usd",
) -> pd.DataFrame:
    _require_cols(strat, [date_col, value_col])
    _require_cols(dca, [date_col, value_col])

    s = strat[[date_col, value_col]].copy().rename(columns={value_col: "strat_value"})
    b = dca[[date_col, value_col]].copy().rename(columns={value_col: "dca_value"})
    j = s.merge(b, on=date_col, how="inner").sort_values(date_col).reset_index(drop=True)

    if len(j) < window_days + 5:
        j["win"] = np.nan
        return j

    win = []
    for i in range(len(j)):
        if i < window_days:
            win.append(np.nan)
            continue
        s_ret = j["strat_value"].iloc[i] / j["strat_value"].iloc[i - window_days] - 1.0
        b_ret = j["dca_value"].iloc[i] / j["dca_value"].iloc[i - window_days] - 1.0
        win.append(float(s_ret > b_ret))

    j["win"] = win
    return j


def sharpe_like(returns: pd.Series, ann_factor: int = 365) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 50:
        return np.nan
    mu = r.mean()
    sd = r.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float((mu / sd) * np.sqrt(ann_factor))


def model_score(win_rate: float, sharpe: float) -> float:
    if np.isnan(win_rate) and np.isnan(sharpe):
        return np.nan
    wr = 0.0 if np.isnan(win_rate) else float(win_rate)
    sh = 0.0 if np.isnan(sharpe) else float(sharpe)
    sh_clip = max(-3.0, min(3.0, sh))
    sh_scaled = (sh_clip + 3.0) / 6.0
    return 0.6 * wr + 0.4 * sh_scaled


def evaluate_strategies(
    joined: pd.DataFrame,
    assumptions: CapitalAssumptions,
    signal_cols: Dict[str, str],
    date_col: str = "date",
    price_col: str = "PriceUSD",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    _require_cols(joined, [date_col, price_col])

    base_mult = pd.Series(1.0, index=joined.index)
    dca_path = simulate_accumulation(joined, assumptions, base_mult, price_col=price_col, date_col=date_col)

    paths: Dict[str, pd.DataFrame] = {"DCA": dca_path}
    rows = []

    for name, col in signal_cols.items():
        if col not in joined.columns:
            continue

        mult = _signal_to_multiplier(joined[col])
        strat_path = simulate_accumulation(joined, assumptions, mult, price_col=price_col, date_col=date_col)

        wins = rolling_win_rate(strat_path, dca_path, window_days=365, date_col=date_col)
        win_rate = float(wins["win"].mean()) if wins["win"].notna().any() else np.nan

        strat_ret = strat_path["portfolio_usd"].pct_change()
        sh = sharpe_like(strat_ret, ann_factor=365)

        spd = float(strat_path["spd"].iloc[-1]) if len(strat_path) else np.nan
        score = model_score(win_rate, sh)

        rows.append(
            {
                "strategy": name,
                "signal_col": col,
                "daily_budget_usd": assumptions.daily_budget_usd,
                "fee_bps": assumptions.fee_bps,
                "slippage_bps": assumptions.slippage_bps,
                "final_cum_usd_spent": float(strat_path["cum_usd_spent"].iloc[-1]),
                "final_cum_btc": float(strat_path["cum_btc"].iloc[-1]),
                "final_portfolio_usd": float(strat_path["portfolio_usd"].iloc[-1]),
                "spd_sats_per_usd": spd,
                "win_rate_1y_windows": win_rate,
                "sharpe_like": sh,
                "model_score": score,
            }
        )

        strat_path = strat_path.merge(
            wins[[date_col, "win"]],
            on=date_col,
            how="left",
        )
        paths[name] = strat_path

    summary = pd.DataFrame(rows).sort_values(["model_score", "win_rate_1y_windows"], ascending=False)
    wins_long = []
    for name, path in paths.items():
        if "win" in path.columns:
            tmp = path[[date_col, "win"]].copy()
            tmp["strategy"] = name
            wins_long.append(tmp)
    wins_long_df = pd.concat(wins_long, ignore_index=True) if wins_long else pd.DataFrame()

    return summary, wins_long_df, paths