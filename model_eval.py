import numpy as np
import pandas as pd

TARGET = "T1"          # change to T2 / T3 to test the others
REFS = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]
B = 432                # 3 days of 10-minute values

# ---------- loading + cleaning ----------
def clean(d):
    """Sort, drop duplicate timestamps, and restore missing timestamps as NaN rows,
    so that 1 row always equals exactly 10 minutes."""
    d = d[~d.index.duplicated()].sort_index()
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="10min"))
    d.index.name = "timestamp"
    return d

def load(name):
    d = pd.read_excel(f"data/{name}_data.xlsx")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = clean(d.set_index("timestamp"))
    if "test_set" in d.columns:
        # restored rows are NaN, and NaN would count as True -> force them to False
        d["test_set"] = d["test_set"].fillna(False).astype(bool)
    return d

def find_col(d, key):
    return [c for c in d.columns if key in c][0]

# ---------- inputs ----------
t = load(TARGET)
cc = pd.read_excel(f"data/{TARGET}_power_curve.xlsx").sort_values("wind_speed")
cx, cy = cc["wind_speed"].to_numpy(float), cc["power"].to_numpy(float)
print("curve ends at wind speed:", cx.max(), "| max power:", cy.max(), "| last power:", cy[-1])

raw = {r: load(r) for r in REFS}
df = pd.DataFrame({r: raw[r]["wind_speed"] for r in REFS})
df["tgt"] = t["wind_speed"]
rowmean = df[REFS].mean(axis=1)
for r in REFS:                                   # fill small gaps in the references
    df[r] = df[r].fillna(rowmean)

# extra physical inputs, all taken from the REFERENCE turbines
rad = np.deg2rad(pd.DataFrame({r: raw[r]["wind_direction"] for r in REFS}))
df["dsin"], df["dcos"] = np.sin(rad).mean(axis=1), np.cos(rad).mean(axis=1)
df["avg"] = df[REFS].mean(axis=1)
df["avg2"] = df["avg"] ** 2
df["temp"] = pd.DataFrame({r: raw[r]["ambient_temperature"] for r in REFS}).mean(axis=1)
df["pres"] = pd.DataFrame({r: raw[r]["air_pressure"] for r in REFS}).mean(axis=1)
df["sd"] = pd.DataFrame({r: raw[r][find_col(raw[r], "stddev")] for r in REFS}).mean(axis=1)
EXTRA = ["avg2", "dsin", "dcos", "temp", "pres", "sd"]
df[EXTRA] = df[EXTRA].interpolate(limit_direction="both")

# only use history BEFORE the official test period
first_test = t.index[t["test_set"]].min()
hist = df[df.index < first_test]
n = len(hist)
starts = [int(n * f) - B for f in (0.55, 0.70, 0.85, 1.0)]
print(f"{TARGET}: history {hist.index[0]} to {hist.index[-1]} ({n} rows), {len(starts)} test outages")

# check how many rows of each fake outage can actually be scored
for s in starts:
    w = hist.iloc[s:s + B]
    print(w.index[0], "->", w.index[-1], "| usable rows:", w["tgt"].notna().sum(), "of", len(w))

# ---------- power helpers ----------
def power(ws):
    return np.interp(np.asarray(ws, float), cx, cy, left=0, right=0)

Z = np.linspace(-2.5, 2.5, 21)                   # expected power over the uncertainty band
W = np.exp(-0.5 * Z ** 2)
W /= W.sum()

def exp_power(ws, sigma):
    x = np.asarray(ws, float)[:, None] + sigma * Z[None, :]
    return (power(np.clip(x, 0, None)) * W[None, :]).sum(axis=1)

# ---------- models: each learns on 'train' and predicts 'test' ----------
def lin_fit(train, cols):
    ok = train[cols + ["tgt"]].dropna()
    A = np.c_[ok[cols].values, np.ones(len(ok))]
    w, *_ = np.linalg.lstsq(A, ok["tgt"].values, rcond=None)
    return lambda X: pd.Series(np.c_[X[cols].values, np.ones(len(X))] @ w, index=X.index)

def m_naive(train, test):
    return pd.Series(train["tgt"].mean(), index=test.index)

def m_best_single(train, test):
    corr = {r: train[r].corr(train["tgt"]) for r in REFS}
    return lin_fit(train, [max(corr, key=corr.get)])(test)

def m_multi(train, test):
    return lin_fit(train, REFS)(test)

def m_multi_stretch(train, test):
    f = lin_fit(train, REFS)
    q = np.linspace(0, 1, 101)
    src = np.quantile(f(train).dropna(), q)
    dst = np.quantile(train["tgt"].dropna(), q)
    return pd.Series(np.interp(f(test).to_numpy(), src, dst), index=test.index)

def m_multi_extra(train, test):
    return lin_fit(train, REFS + EXTRA)(test)

def best_lag(x, y, max_lag=12):
    # try shifts of up to +-12 steps (+-2 hours), keep the one with the highest correlation
    c = {k: y.corr(x.shift(k)) for k in range(-max_lag, max_lag + 1)}
    return max(c, key=c.get)

def m_multi_lag(train, test):
    lags = {r: best_lag(train[r], train["tgt"]) for r in REFS}   # learned on training data only
    sh = pd.DataFrame({r: hist[r].shift(lags[r]) for r in REFS}, index=hist.index)
    sh["tgt"] = hist["tgt"]
    return lin_fit(sh.loc[train.index], REFS)(sh.loc[test.index])
def m_sector(train, test, nsec=8):
    # wind direction of the references, split into 8 compass sectors
    ang = np.degrees(np.arctan2(hist["dsin"], hist["dcos"])) % 360
    sec = (((ang + 180 / nsec) % 360) // (360 / nsec)).astype(int)
    out = lin_fit(train, ["avg"])(test)              # fallback: one formula for all directions
    for k in range(nsec):
        tr = train[sec.loc[train.index] == k]
        idx = test.index[sec.loc[test.index] == k]
        if tr["tgt"].notna().sum() >= 200 and len(idx) > 0:
            out.loc[idx] = lin_fit(tr, ["avg"])(test.loc[idx])
    return out

# name -> (model, use expected power over the uncertainty band?)
MODELS = {
    "D: naive (training mean)": (m_naive, False),
    "A: best single turbine": (m_best_single, False),
    "C: multiple regression": (m_multi, False),
    "F: C + stretch": (m_multi_stretch, False),
    "G: C + expected power": (m_multi, True),
    "H: C + physics inputs": (m_multi_extra, False),
    "I: H + expected power": (m_multi_extra, True),
    "J: C + lag": (m_multi_lag, False),
    "K: G + lag": (m_multi_lag, True),
    "L: sectors + expected power": (m_sector, True),
}

# ---------- scoring ----------
def evaluate(model, jensen):
    A_, P_, PA, PP, dev = [], [], [], [], []
    for s in starts:
        train, test = hist.iloc[:s], hist.iloc[s:s + B]
        pred = model(train, test).clip(lower=0)
        ok = test["tgt"].notna() & pred.notna()
        a, p = test["tgt"][ok], pred[ok]
        pa = power(a)
        if jensen:
            sigma = (train["tgt"] - model(train, train)).std()
            pp = exp_power(p, sigma)
        else:
            pp = power(p)
        A_.append(a); P_.append(p); PA.append(pa); PP.append(pp)
        dev.append(100 * (pp.sum() - pa.sum()) / pa.sum() if pa.sum() > 0 else np.nan)
    a, p = pd.concat(A_), pd.concat(P_)
    pa, pp = np.concatenate(PA), np.concatenate(PP)
    return {
        "RMSEws": np.sqrt(((p - a) ** 2).mean()),
        "MAEws": (p - a).abs().mean(),
        "wsBias": (p - a).mean(),
        "RMSEkW": np.sqrt(((pp - pa) ** 2).mean()),
        "MAEkW": np.abs(pp - pa).mean(),
        "devMean%": np.nanmean(np.abs(dev)),
        "devWorst%": np.nanmax(np.abs(dev)),
        "bias%": np.nanmean(dev),
        "spread": p.std() / a.std(),
    }

rows = {name: evaluate(f, j) for name, (f, j) in MODELS.items()}
print()
print(pd.DataFrame(rows).T.round(3).to_string())