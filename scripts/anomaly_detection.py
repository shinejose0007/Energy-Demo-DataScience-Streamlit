# scripts/anomaly_detection.py
"""
Robust anomaly detection for time-series consumption data.

Provides:
- seasonal_mad_zscore(...) -> (df_out, summary)
- isolation_forest_detector(...) -> df_out (with is_anomaly_iforest)
- combined_anomalies(...) -> dict with mad_df, iforest_df, joined, summary
"""
from typing import Tuple, Dict
import pandas as pd
import numpy as np

def _mad(x):
    return np.median(np.abs(x - np.median(x)))

def seasonal_mad_zscore(
    df: pd.DataFrame,
    value_col: str = "consumption_kwh",
    window_days: int = 7,
    z_thresh: float = 3.0,
    top_n: int = 20,
) -> Tuple[pd.DataFrame, Dict]:
    df = df.copy()
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must contain 'datetime' column")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df["hour"] = df["datetime"].dt.hour

    out_frames = []
    # for each hour-of-day compute rolling median and MAD across recent days (window_days)
    for h in range(24):
        sub = df.loc[df["hour"] == h].copy().reset_index(drop=True)
        if sub.empty:
            continue
        window = max(1, int(window_days))
        med = sub[value_col].rolling(window=window, min_periods=1).median()
        mad = sub[value_col].rolling(window=window, min_periods=1).apply(_mad, raw=True).fillna(0.0)
        mad_as_std = mad * 1.4826  # approx std
        z = (sub[value_col] - med) / mad_as_std.replace(0, np.nan)
        z = z.fillna(0.0)
        sub = sub.assign(median_hour=med.values, mad=mad.values, mad_as_std=mad_as_std.values, z_score=z.values)
        out_frames.append(sub)

    if not out_frames:
        # no data for any hour
        df_out = df.copy()
        df_out["z_score"] = 0.0
        df_out["is_anomaly"] = False
        summary = {"n_rows": 0, "n_flagged": 0, "z_thresh": float(z_thresh), "top_n_rows": pd.DataFrame()}
        return df_out, summary

    df_out = pd.concat(out_frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    df_out["is_anomaly"] = df_out["z_score"].abs() > float(z_thresh)
    total_flagged = int(df_out["is_anomaly"].sum())
    topn = df_out.reindex(df_out["z_score"].abs().sort_values(ascending=False).index).head(top_n)

    summary = {
        "n_rows": len(df_out),
        "n_flagged": total_flagged,
        "z_thresh": float(z_thresh),
        "top_n_rows": topn,
    }
    return df_out, summary

def isolation_forest_detector(df: pd.DataFrame, value_col: str = "consumption_kwh", contamination: float = 0.01) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    try:
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        raise ImportError("scikit-learn required for isolation_forest_detector. Install scikit-learn") from e

    X = df[[value_col]].fillna(0.0).values
    if len(X) < 5:
        df["is_anomaly_iforest"] = False
        return df
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X)
    df["is_anomaly_iforest"] = preds == -1
    return df

def combined_anomalies(
    df: pd.DataFrame,
    value_col: str = "consumption_kwh",
    window_days: int = 7,
    z_thresh: float = 3.0,
    top_n: int = 20,
    contamination: float = 0.01,
) -> Dict:
    mad_df, mad_summary = seasonal_mad_zscore(df, value_col=value_col, window_days=window_days, z_thresh=z_thresh, top_n=top_n)
    try:
        if_df = isolation_forest_detector(df, value_col=value_col, contamination=contamination)
    except Exception:
        if_df = None

    joined = mad_df.copy()
    if if_df is not None and "is_anomaly_iforest" in if_df.columns:
        joined = joined.merge(if_df[["datetime", "is_anomaly_iforest"]], on="datetime", how="left")
        joined["is_anomaly_iforest"] = joined["is_anomaly_iforest"].fillna(False)
    else:
        joined["is_anomaly_iforest"] = False

    joined["is_anomaly_combined"] = joined["is_anomaly"] | joined["is_anomaly_iforest"]

    summary = {
        "mad_n_flagged": int(mad_df["is_anomaly"].sum()),
        "iforest_n_flagged": int(if_df["is_anomaly_iforest"].sum()) if (if_df is not None and "is_anomaly_iforest" in if_df.columns) else 0,
        "combined_n_flagged": int(joined["is_anomaly_combined"].sum()),
        "mad_summary": mad_summary,
    }
    return {"mad_df": mad_df, "iforest_df": if_df, "joined": joined, "summary": summary}
