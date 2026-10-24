import numpy as np
import pandas as pd
from outage_model import predict_wind

_SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _sector_index(direction, index):
    s = (direction.reindex(index) + 22.5) // 45 % 8
    return pd.Series([_SECTORS[int(x)] if pd.notna(x) else None for x in s], index=index)


def reference_direction(raw_dirs):
    """raw_dirs: DataFrame of wind directions (degrees), one column per reference turbine."""
    rad = np.deg2rad(raw_dirs)
    return np.rad2deg(np.arctan2(np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1))) % 360


def predict_wind_corrected(df, refs, train_mask, out_idx, direction=None, min_rows=200):
    """Like predict_wind, plus a per-direction bias correction learned on training rows only.
    Without a direction series it falls back to predict_wind."""
    pred, info = predict_wind(df, refs, train_mask, out_idx)
    if direction is None:
        info["direction_correction"] = {}
        return pred, info
    tr_idx = df.index[train_mask & df["tgt"].notna()]
    pred_tr, _ = predict_wind(df, refs, train_mask, tr_idx)
    resid = (pred_tr - df.loc[tr_idx, "tgt"]).dropna()
    stats = resid.groupby(_sector_index(direction, resid.index)).agg(["size", "mean"])
    corr = {s: float(r["mean"]) for s, r in stats.iterrows() if r["size"] >= min_rows}
    shift = _sector_index(direction, pred.index).map(corr).fillna(0.0).to_numpy()
    info["direction_correction"] = corr
    return (pred - shift).clip(lower=0), info