import numpy as np
import pandas as pd

TARGET = "T1"
REFS = ["H1", "H2", "H3", "H4", "H5", "O1", "O2"]

def load(name):
    d = pd.read_excel(f"data/{name}_data.xlsx")
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    return d.set_index("timestamp").sort_index()

t = load(TARGET)
raw = {r: load(r) for r in REFS}

for name, d in {TARGET: t, **raw}.items():
    ws = d["wind_speed"]
    print(name,
          "| NaN:", ws.isna().sum(),
          f"({100*ws.isna().mean():.1f}%)",
          "| min/max:", ws.min(), ws.max(),
          "| duplicate times:", d.index.duplicated().sum())

full = pd.date_range(t.index.min(), t.index.max(), freq="10min")
print("missing timestamps in T1:", len(full.difference(t.index)))

full = pd.date_range(t.index.min(), t.index.max(), freq="10min")
missing = full.difference(t.index)
print("missing timestamps:", len(missing))
print("first:", missing.min(), "| last:", missing.max())