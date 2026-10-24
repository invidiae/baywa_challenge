"""Evaluate the final model on the SHOWN test days (4-9 Sep), where real measurements exist.
The model learns only from training rows (never from 1-9 Sep), then predicts 4-9 Sep.
Run from the folder that contains the 'data' folder and outage_model.py:   py eval_shown.py
"""
import numpy as np
import pandas as pd
from outage_model import predict_wind, expected_power, to_power

REFS = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]
TARGETS = ["T1", "T2", "T3"]

# --- choose what counts as training data ---
# "before_outage": only rows before 1 Sep (strict, matches "learn from the past")
# "all_train":     every row with test_set = False (also includes 10-30 Sep)
TRAIN_MODE = "before_outage"

SHOWN_START = pd.Timestamp("2026-09-04")
SHOWN_END = pd.Timestamp("2026-09-10")        # exclusive


def load(name):
    d = pd.read_excel(f"data/{name}_data.xlsx")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()


rows = []
tot_act = tot_pred = 0.0
for T in TARGETS:
    t = load(T)
    cc = pd.read_excel(f"data/{T}_power_curve.xlsx").sort_values("wind_speed")
    curve = (cc["wind_speed"].to_numpy(float), cc["power"].to_numpy(float))

    idx = pd.date_range(t.index.min(), t.index.max(), freq="10min")   # restores missing rows as NaN
    df = pd.DataFrame({r: load(r)["wind_speed"] for r in REFS}).reindex(idx)
    df["tgt"] = t["wind_speed"].reindex(idx)
    test = t["test_set"].reindex(idx).fillna(False).astype(bool)

    if TRAIN_MODE == "before_outage":
        train_mask = df.index < test.index[test].min()      # before the first test row
    else:
        train_mask = ~test                                  # all rows outside the test period

    assert not (train_mask & test).any(), "training rows overlap the test period!"

    out = df.index[(df.index >= SHOWN_START) & (df.index < SHOWN_END)]
    pred, info = predict_wind(df, REFS, train_mask, out)

    act = df.loc[out, "tgt"]
    ok = act.notna() & pred.notna()
    a, p = act[ok], pred[ok]
    pa = to_power(a, curve)                                  # power from the real wind
    pp = expected_power(p, curve, info["sigma"])             # power from the predicted wind

    e_act, e_pred = pa.sum() / 6, pp.sum() / 6               # kW x 10 min -> kWh
    tot_act += e_act
    tot_pred += e_pred
    rows.append({
        "Turbine": T,
        "Rows scored": len(a),
        "RMSE wind (m/s)": np.sqrt(((p - a) ** 2).mean()),
        "MAE wind (m/s)": (p - a).abs().mean(),
        "Wind bias (m/s)": (p - a).mean(),
        "RMSE power (kW)": np.sqrt(((pp - pa) ** 2).mean()),
        "MAE power (kW)": (pp - pa).abs().mean(),
        "Energy actual (kWh)": e_act,
        "Energy predicted (kWh)": e_pred,
        "Energy deviation (%)": 100 * (e_pred - e_act) / e_act,
    })

print(f"Training mode: {TRAIN_MODE}")
print(f"Test period:   {SHOWN_START:%d %b} to {(SHOWN_END - pd.Timedelta(minutes=10)):%d %b %Y}\n")
print(pd.DataFrame(rows).set_index("Turbine").round(2).T.to_string())
print(f"\nAll three turbines: actual {tot_act:,.0f} kWh | predicted {tot_pred:,.0f} kWh | "
      f"deviation {100 * (tot_pred - tot_act) / tot_act:+.1f}%")