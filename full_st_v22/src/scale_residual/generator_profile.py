from __future__ import annotations
import pandas as pd

LB_TO_KG = 0.4535924

# Frozen compatibility profile used by the manuscript experiments.
# Cost = fuel price ($/MMBtu) * representative heat rate (MMBtu/MWh).
# EF   = representative heat rate * fuel CO2 factor (lb/MMBtu) * lb->kg.
PROFILE = {
    "OIL_CT20":   dict(pmax=20.0,  cost=10.3494*13.114, ef=13.114*160*LB_TO_KG),
    "COAL_76":    dict(pmax=76.0,  cost=2.11399*13.270, ef=13.270*210*LB_TO_KG),
    "GAS_CC355":  dict(pmax=355.0, cost=3.88722*7.222,   ef=7.222*118*LB_TO_KG),
    "GAS_CT55":   dict(pmax=55.0,  cost=3.88722*13.125,  ef=13.125*118*LB_TO_KG),
    "OIL_ST12":   dict(pmax=12.0,  cost=10.3494*17.340, ef=17.340*170*LB_TO_KG),
    "COAL_155":   dict(pmax=155.0, cost=2.11399*11.446, ef=11.446*210*LB_TO_KG),
    "COAL_350":   dict(pmax=350.0, cost=2.11399*12.106, ef=12.106*210*LB_TO_KG),
    "NUCLEAR_400":dict(pmax=400.0, cost=0.81035*10.0,    ef=0.0),
    "HYDRO_50":   dict(pmax=50.0,  cost=0.0,             ef=0.0),
}


def build_paper_compat_profile() -> pd.DataFrame:
    rows: list[dict] = []
    def add(bus: int, typ: str, n: int):
        par = PROFILE[typ]
        for k in range(1, n+1):
            rows.append(dict(uid=f"{bus}_{typ}_{k}", bus=bus, typ=typ, **par))
    # Area 1
    add(101,"OIL_CT20",2); add(101,"COAL_76",2)
    add(102,"OIL_CT20",2); add(102,"COAL_76",2)
    add(107,"GAS_CC355",1)
    add(113,"GAS_CT55",4)
    add(115,"OIL_ST12",2); add(115,"COAL_155",1)
    add(116,"COAL_155",1)
    add(118,"GAS_CC355",1)
    add(123,"COAL_155",1); add(123,"COAL_350",1); add(123,"GAS_CT55",3)
    add(121,"NUCLEAR_400",1)
    add(122,"HYDRO_50",6)
    # Area 2
    add(201,"OIL_CT20",2); add(201,"COAL_76",1); add(201,"HYDRO_50",1)
    add(202,"OIL_CT20",2); add(202,"COAL_76",2)
    add(207,"GAS_CT55",2)
    add(213,"GAS_CC355",1); add(213,"GAS_CT55",2)
    add(215,"GAS_CT55",2); add(215,"HYDRO_50",3)
    add(216,"COAL_155",1)
    add(218,"GAS_CC355",1)
    add(221,"GAS_CC355",1)
    add(223,"COAL_155",2); add(223,"COAL_350",1); add(223,"GAS_CT55",3)
    # Area 3
    add(301,"OIL_CT20",2); add(301,"GAS_CT55",2)
    add(302,"OIL_CT20",2); add(302,"GAS_CT55",2)
    add(307,"GAS_CT55",2)
    add(313,"GAS_CC355",1)
    add(315,"OIL_ST12",5); add(315,"GAS_CT55",3)
    add(316,"COAL_155",1)
    add(318,"GAS_CC355",1)
    add(321,"GAS_CC355",1)
    add(322,"GAS_CT55",2); add(322,"HYDRO_50",4)
    add(323,"GAS_CC355",2)
    return pd.DataFrame(rows)


def wind_units() -> pd.DataFrame:
    return pd.DataFrame([
        dict(uid="309_WIND_1", bus=309, typ="WIND", pmax=148.3, cost=0.0, ef=0.0),
        dict(uid="317_WIND_1", bus=317, typ="WIND", pmax=799.1, cost=0.0, ef=0.0),
        dict(uid="303_WIND_1", bus=303, typ="WIND", pmax=847.0, cost=0.0, ef=0.0),
        dict(uid="122_WIND_1", bus=122, typ="WIND", pmax=713.5, cost=0.0, ef=0.0),
    ])


def aggregate_bus_type(units: pd.DataFrame, bus: pd.DataFrame) -> pd.DataFrame:
    area = bus.set_index("Bus ID")["Area"].to_dict()
    units = units.copy()
    units["area"] = units["bus"].map(area).astype(int)
    out = (units.groupby(["bus","typ","area"], as_index=False)
           .agg(pmax=("pmax","sum"), cost=("cost","first"), ef=("ef","first")))
    out["gid"] = range(len(out))
    return out
