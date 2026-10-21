import numpy as np
import pandas as pd
from scale_residual.network import build_dc_network
from scale_residual.hidden_state import project_generation

def test_projection_respects_balance_and_monthly_categories():
    bus=pd.DataFrame({"Bus ID":[1,2],"MW Load":[50.,50.],"Area":[1,2]})
    branch=pd.DataFrame({"From Bus":[1],"To Bus":[2],"X":[0.1],"Cont Rating":[1000.]})
    net=build_dc_network(bus,branch)
    gen=pd.DataFrame({"bus":[1,2],"typ":["A","B"],"area":[1,2],"pmax":[100.,100.],"cost":[1.,1.],"ef":[500.,100.]})
    Lreg=np.array([[40.,60.],[60.,40.],[50.,50.]])
    Pprior=np.array([[50.,50.],[50.,50.],[50.,50.]])
    cats=[dict(area=1,typ="A",ids=np.array([0]),energy=180.), dict(area=2,typ="B",ids=np.array([1]),energy=120.)]
    P,Lbus,audit=project_generation(net,gen,Lreg,Pprior,cats)
    assert audit.hour_balance_max_mw < 1e-7
    assert audit.monthly_category_max_mwh < 1e-7
    assert audit.max_line_ratio <= 1+1e-9
    assert abs(P[:,0].sum()-180.) < 1e-7
    assert abs(P[:,1].sum()-120.) < 1e-7
