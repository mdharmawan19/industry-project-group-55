"""Victoria in Future 2023 SA2 projections (population, dwellings)."""
import numpy as np
import pandas as pd

def load_vif_sa2(path, sheet="Total_Dwellings", prefix="DWELL", min_base=100):
    """One row per SA2 with VIF values for 2021/26/31/36 plus compound annual growth.
    Growth is left blank where the base count is below min_base (non-residential SA2s)."""
    df = pd.read_excel(path, sheet_name=sheet, header=9)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    df = df[df["Region Type"].astype(str).str.strip() == "SA2"].copy()
    years = {c: int(float(c)) for c in df.columns if c.replace(".", "").isdigit()}
    df = df.rename(columns={c: f"{prefix}_{y}" for c, y in years.items()})
    df["SA2_CODE21"] = df["SA2 code"].astype(int).astype(str)
    out = df[["SA2_CODE21", "GCCSA"] + [f"{prefix}_{y}" for y in sorted(years.values())]].copy()
    for a, b in [(2021, 2026), (2026, 2031)]:
        base, end = out[f"{prefix}_{a}"], out[f"{prefix}_{b}"]
        g = (end / base) ** (1 / (b - a)) - 1
        out[f"{prefix}_GROWTH_{a}_{b}"] = g.where(base >= min_base, np.nan)
    return out.reset_index(drop=True)
