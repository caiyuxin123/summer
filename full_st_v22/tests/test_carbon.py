import numpy as np
import pandas as pd
from scale_residual.network import build_dc_network, generator_incidence
from scale_residual.carbon import nodal_cef
from scale_residual.aggregate import aggregate_state

def test_single_source_chain_cef():
    bus=pd.DataFrame({"Bus ID":[1,2],"MW Load":[1.,100.],"Area":[1,2]})
    branch=pd.DataFrame({"From Bus":[1],"To Bus":[2],"X":[0.1],"Cont Rating":[1000.]})
    net=build_dc_network(bus,branch)
    gen=pd.DataFrame({"bus":[1],"typ":["THERM"],"area":[1],"pmax":[200.],"cost":[1.],"ef":[500.]})
    p=np.array([100.]); load=np.array([0.,100.]); flow=np.array([100.])
    x=nodal_cef(net,gen,p,load,flow)
    assert np.allclose(x,[500.,500.])

def test_single_node_per_area_has_zero_spatial_residual():
    bus=pd.DataFrame({"Bus ID":[1,2],"MW Load":[50.,50.],"Area":[1,2]})
    branch=pd.DataFrame({"From Bus":[1],"To Bus":[2],"X":[0.1],"Cont Rating":[1000.]})
    net=build_dc_network(bus,branch)
    gen=pd.DataFrame({"bus":[1,2],"typ":["A","B"],"area":[1,2],"pmax":[100.,100.],"cost":[1.,2.],"ef":[500.,100.]})
    Lreg=np.array([[50.,50.]])
    Lbus=np.array([[50.,50.]])
    P=np.array([[100.,0.]])
    F=np.array([[50.]])
    X=np.array([[500.,500.]])
    out=aggregate_state(net,gen,Lreg,Lbus,P,F,X)
    assert np.max(np.abs(out.delta_spatial)) < 1e-10
