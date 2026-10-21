import numpy as np
import pandas as pd
from scale_residual.network import build_dc_network, regional_to_nodal_load
from scale_residual.dispatch import dc_opf_timeseries
from scale_residual.carbon import nodal_cef_timeseries
from scale_residual.aggregate import aggregate_state
from scale_residual.priors import generation_categories
from scale_residual.hidden_state import project_generation


def test_end_to_end_scale_residual_correction_smoke():
    bus=pd.DataFrame({"Bus ID":[1,2,3],"MW Load":[60.,40.,100.],"Area":[1,1,2]})
    branch=pd.DataFrame({"From Bus":[1,2,1],"To Bus":[2,3,3],"X":[0.2,0.2,0.4],"Cont Rating":[500.,500.,500.]})
    net=build_dc_network(bus,branch)
    gen=pd.DataFrame({
        "bus":[1,3],"typ":["coal","gas"],"area":[1,2],
        "pmax":[250.,250.],"cost":[20.,40.],"ef":[850.,400.]
    })
    Lreg=np.array([[100.,80.],[120.,90.],[90.,110.]],float)
    Lbus=regional_to_nodal_load(net,Lreg)
    P,F=dc_opf_timeseries(net,gen,Lbus)
    X=nodal_cef_timeseries(net,gen,P,Lbus,F)
    ref=aggregate_state(net,gen,Lreg,Lbus,P,F,X)
    cats=generation_categories(gen,P)
    Pest,Lbest,audit=project_generation(net,gen,Lreg,P.copy(),cats)
    Fest=Pest@(net.H@np.array([[1,0],[0,0],[0,1]],float)).T-(net.H@Lbest.T).T
    Xest=nodal_cef_timeseries(net,gen,Pest,Lbest,Fest)
    est=aggregate_state(net,gen,Lreg,Lbest,Pest,Fest,Xest)
    corrected=ref.direct_month+(est.fine_month-est.direct_month)
    assert np.max(np.abs(corrected-ref.fine_month)) < 1e-7
    assert audit.max_line_ratio <= 1+1e-9
