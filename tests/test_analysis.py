import pandas as pd
import pytest

from src.analysis import eslestirilmis_fark


def test_eslestirilmis_fark_elle_hesap():
    df = pd.DataFrame({
        "veri": "x", "oran": 0.6, "seed": [0, 1, 2] * 2,
        "model": ["A"] * 3 + ["B"] * 3,
        "F1": [0.5, 0.6, 0.7, 0.6, 0.6, 0.65]})
    r = eslestirilmis_fark(df, "A", "F1").iloc[0]
    assert r["model"] == "B"
    assert r["fark_ort"] == pytest.approx((0.1 + 0.0 - 0.05) / 3)
    assert r["daha_iyi_seed"] == 1 and r["n_seed"] == 3
    # düşük-iyi metrikte yön tersine döner
    df["RMSE"] = df["F1"]
    r2 = eslestirilmis_fark(df, "A", "RMSE", yuksek_iyi=False).iloc[0]
    assert r2["daha_iyi_seed"] == 1
