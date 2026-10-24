"""Train on ALL training rows (test_set == False) and predict T1, T2, T3
for the whole test period (1-9 Sep).
 - 1-3 Sep: the real outage (measurements are NaN)  -> energy loss estimate
 - 4-9 Sep: real measurements exist                  -> accuracy check
Run:   py ver1.py
"""
import numpy as np
import pandas as pd
from outage_model import expected_power, to_power
from direction_correction import predict_wind_corrected, reference_direction

REFS = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]
TARGETS = ["T1", "T2", "T3"]
OUT_FILE = "predictions_T1_T2_T3.csv"
USE_DIRECTION_CORRECTION = True      # False = plain model, True = with per-direction correction


def load(name):
    d = pd.read_excel(f"data/{name}_data.xlsx")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()


dirs = pd.DataFrame({r: load(r)["wind_direction"] for r in REFS})

frames, summary = [], []
for T in TARGETS:
    t = load(T)
    cc = pd.read_excel(f"data/{T}_power_curve.xlsx").sort_values("wind_speed")
    curve = (cc["wind_speed"].to_numpy(float), cc["power"].to_numpy(float))

    idx = pd.date_range(t.index.min(), t.index.max(), freq="10min")   # missing rows become NaN
    df = pd.DataFrame({r: load(r)["wind_speed"] for r in REFS}).reindex(idx)
    df["tgt"] = t["wind_speed"].reindex(idx)
    test = t["test_set"].reindex(idx).fillna(False).astype(bool)
    direction = reference_direction(dirs.reindex(idx)) if USE_DIRECTION_CORRECTION else None

    # ---- train on everything that is NOT in the test period ----
    train_mask = ~test
    assert not (train_mask & test).any(), "training rows overlap the test period!"

    # ---- predict every row of the test period ----
    out = df.index[test]
    pred, info = predict_wind_corrected(df, REFS, train_mask, out, direction)
    pred = pred.interpolate(limit_direction="both")
    power_pred = expected_power(pred, curve, info["sigma"])           # kW per 10-minute step
    print(f"{T} direction correction (m/s):",
          {k: round(v, 2) for k, v in info["direction_correction"].items()})

    actual = df.loc[out, "tgt"]
    is_outage = actual.isna()                                          # NaN = lost connection

    frames.append(pd.DataFrame({
        f"{T}_wind_pred": pred,
        f"{T}_wind_actual": actual,
        f"{T}_power_pred_kW": power_pred,
        f"{T}_is_outage": is_outage,
    }))

    row = {
        "Turbine": T,
        "Training rows used": int((train_mask & df["tgt"].notna()).sum()),
        "Outage hours (NaN)": is_outage.sum() / 6,
        "Predicted loss in outage (kWh)": power_pred[is_outage].sum() / 6,
    }

    # ---- accuracy on the days where real measurements exist ----
    ok = actual.notna() & pred.notna()
    a, p = actual[ok], pred[ok]
    pa = to_power(a, curve)
    pp = expected_power(p, curve, info["sigma"])
    e_act, e_pred = pa.sum() / 6, pp.sum() / 6
    row.update({
        "Scored rows": len(a),
        "RMSE wind (m/s)": np.sqrt(((p - a) ** 2).mean()),
        "MAE wind (m/s)": (p - a).abs().mean(),
        "Wind bias (m/s)": (p - a).mean(),
        "RMSE power (kW)": np.sqrt(((pp - pa) ** 2).mean()),
        "MAE power (kW)": (pp - pa).abs().mean(),
        "Energy actual (kWh)": e_act,
        "Energy predicted (kWh)": e_pred,
        "Energy deviation (%)": 100 * (e_pred - e_act) / e_act,
    })
    summary.append(row)

result = pd.concat(frames, axis=1)
result.index.name = "timestamp"
result.to_csv(OUT_FILE)

print(f"\nDirection correction: {'ON' if USE_DIRECTION_CORRECTION else 'OFF'}")
print(f"Test period: {result.index.min():%d %b %H:%M} to {result.index.max():%d %b %H:%M %Y}")
print(pd.DataFrame(summary).set_index("Turbine").round(2).T.to_string())
print(f"\nSaved all predictions to {OUT_FILE}")