from operations import *
from conversions import *

def get_spatial_jacobian(gs, screws):
    Ads = []
    for i in range(len(gs)):
        Ads.append(se3_derepr(Ad(gs[i], screws[i])))
    
    Js = np.vstack(Ads).T
    return Js

def get_body_jacobian(g, Js=None, gs=None, screws=None):
    if Js is None:
        Js = get_spatial_jacobian(gs, screws)
    
    Ad_invs = []
    for j in Js.T:
        Ad_invs.append(se3_derepr(Ad_inv(g, se3_repr(j))))
    
    Jb = np.vstack(Ad_invs).T
    return Jb

def get_world_jacobian(g, Js=None, gs=None, screws=None):
    if Js is None:
        Js = get_spatial_jacobian(gs, screws)
    
    TeRgs = []
    for j in Js.T:
        TeRgs.append(se3_derepr(right_lifted_action(g, se3_repr(j))))
    
    Jw = np.vstack(TeRgs).T
    return Jw
        
Js = get_spatial_jacobian([SE3_from_xyz_rpy(0, 0, 0, 0, 0, 0), SE3_from_xyz_rpy(1, 0, 0, 0, 0, 0)], [se3_repr([0, 0, 0, 0, 0, 1]), se3_repr([0, 0, 0, 0, 0, -1])])
print(Js)

Jb = get_body_jacobian(SE3_from_xyz_rpy(0.5, 0.5, 0, 0, 0, 0), Js)
print(Jb)

Jw = get_world_jacobian(SE3_from_xyz_rpy(0.5, 0.5, 0, 0, 0, 0), Js)
print(Jw)