import numpy as np
import pandas as pd
from scale_residual.network import build_dc_network, regional_to_nodal_load

def fixture():
    bus=pd.DataFrame({"Bus ID":[1,2,3],"MW Load":[50.,50.,100.],"Area":[1,1,2]})
    branch=pd.DataFrame({"From Bus":[1,2],"To Bus":[2,3],"X":[0.1,0.1],"Cont Rating":[1000.,1000.]})
    return bus,branch

def test_ptdf_balanced_injection():
    bus,branch=fixture(); net=build_dc_network(bus,branch,slack_bus=1)
    inj=np.array([10.,0.,-10.])
    f=net.H@inj
    assert np.all(np.isfinite(f))
    assert abs(f[1]-10.) < 1e-8

def test_regional_load_conservation():
    bus,branch=fixture(); net=build_dc_network(bus,branch)
    reg=np.array([[100.,80.],[120.,90.]])
    nod=regional_to_nodal_load(net,reg)
    assert np.allclose(nod[:,net.area_of_bus==1].sum(1),reg[:,0])
    assert np.allclose(nod[:,net.area_of_bus==2].sum(1),reg[:,1])
