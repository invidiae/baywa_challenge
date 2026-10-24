"""Outage model (method K): lag-aligned regression + expected power.

Steps
1. Each reference turbine is shifted in time by the lag that matches the affected turbine best.
2. A multiple regression on the shifted references predicts the missing wind speed.
3. Wind -> power with the power curve, averaged over the typical prediction error
   ("expected power"), because the power curve is not a straight line.
4. Energy (kWh) = sum of power (kW) x 10 min = sum / 6.

Expected input: a DataFrame `df` on a complete 10-minute index with the reference
columns (names in `refs`) and a column "tgt" (affected turbine, NaN during the outage).
"""
import numpy as np
import pandas as pd

Z = np.linspace(-2.5, 2.5, 21)      # grid over the prediction-error band
W = np.exp(-0.5 * Z ** 2)
W /= W.sum()


# ---------- power ----------
def to_power(ws, curve):
    cx, cy = curve
    ws = pd.Series(ws)
    return pd.Series(np.interp(ws.to_numpy(float), cx, cy, left=0, right=0), index=ws.index)


def expected_power(ws, curve, sigma):
    """Average power over the wind-speed uncertainty (normal band, std = sigma)."""
    ws = pd.Series(ws)
    sigma = 0.0 if not np.isfinite(sigma) else sigma
    x = ws.to_numpy(float)[:, None] + sigma * Z[None, :]
    cx, cy = curve
    p = np.interp(np.clip(x, 0, None), cx, cy, left=0, right=0)
    return pd.Series((p * W[None, :]).sum(axis=1), index=ws.index)


# ---------- wind prediction ----------
def best_lag(x, y, max_lag=12):
    """Shift of x (in 10-min steps, +-max_lag) that correlates best with y."""
    c = {k: y.corr(x.shift(k)) for k in range(-max_lag, max_lag + 1)}
    c = {k: v for k, v in c.items() if np.isfinite(v)}
    return max(c, key=c.get) if c else 0


def predict_wind(df, refs, fit_mask, pred_idx, max_lag=12):
    """Learn tgt from refs on the rows where fit_mask is True, predict at pred_idx."""
    rm = df[refs].mean(axis=1)
    X = df[refs].apply(lambda c: c.fillna(rm))            # small reference gaps
    y = df["tgt"].where(fit_mask)                         # only training rows are used to learn
    lags = {r: best_lag(X[r], y, max_lag) for r in refs}
    S = pd.DataFrame({r: X[r].shift(lags[r]) for r in refs})
    ok = fit_mask & S.notna().all(axis=1) & df["tgt"].notna()
    A = np.c_[S[ok].values, np.ones(ok.sum())]
    w, *_ = np.linalg.lstsq(A, df.loc[ok, "tgt"].values, rcond=None)
    sigma = float(np.std(df.loc[ok, "tgt"].values - A @ w))
    Sp = S.loc[pred_idx]
    pred = pd.Series(np.c_[Sp.values, np.ones(len(Sp))] @ w, index=pred_idx).clip(lower=0)
    info = {"lags": lags, "weights": dict(zip(refs, w[:-1])), "b": float(w[-1]), "sigma": sigma}
    return pred, info


# ---------- validation ----------
def backtest(hist, refs, curve, factor=1, B=432, fracs=(0.55, 0.70, 0.85, 1.0)):
    """Walk-forward test: hide B rows, learn only from the rows before, compare with the truth.
    `hist` must be the history before the outage (contiguous 10-minute rows).
    Returns (metrics dict, measured wind, predicted wind)."""
    n = len(hist)
    B = min(B, n // 5)
    A_, P_, PA, PP, dev = [], [], [], [], []
    for f in fracs:
        s = int(n * f) - B
        fm = pd.Series(np.arange(n) < s, index=hist.index)
        idx = hist.index[s:s + B]
        pred, info = predict_wind(hist, refs, fm, idx)
        a = hist.loc[idx, "tgt"]
        ok = a.notna() & pred.notna()
        a, p = a[ok], pred[ok]
        if len(a) == 0:
            continue
        pa = to_power(a, curve) * factor
        pp = expected_power(p, curve, info["sigma"]) * factor
        A_.append(a); P_.append(p); PA.append(pa); PP.append(pp)
        if pa.sum() > 0:
            dev.append(100 * (pp.sum() - pa.sum()) / pa.sum())
    a, p = pd.concat(A_), pd.concat(P_)
    pa, pp = pd.concat(PA), pd.concat(PP)
    dev = dev or [np.nan]
    m = {
        "RMSE wind (m/s)": float(np.sqrt(((p - a) ** 2).mean())),
        "MAE wind (m/s)": float((p - a).abs().mean()),
        "RMSE power (kW)": float(np.sqrt(((pp - pa) ** 2).mean())),
        "MAE power (kW)": float((pp - pa).abs().mean()),
        "Energy deviation, mean abs (%)": float(np.nanmean(np.abs(dev))),
        "Energy deviation, worst (%)": float(np.nanmax(np.abs(dev))),
        "Energy bias (%)": float(np.nanmean(dev)),
    }
    return m, a, p