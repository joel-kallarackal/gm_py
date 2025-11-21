from operations import *
from conversions import *

MatNx4x4 = np.ndarray # (N, 4, 4)
Mat6xN = np.ndarray   # (6, N)

def get_spatial_jacobian(gs: MatNx4x4, screws: MatNx4x4) -> Mat6xN:
    """
    Args:
        gs (MatNx4x4, optional): SE(3) configuration of each joints
        screws (MatNx4x4, optional): screw axis for joints

    Returns:
        Mat6xN: Spatial Jacobian
    """
    Ads = []
    for i in range(len(gs)):
        Ads.append(se3_derepr(Ad(gs[i], screws[i])))
    
    Js = np.vstack(Ads).T
    return Js

def get_body_jacobian(g: Mat4x4, Js: Mat6xN = None, gs: MatNx4x4 = None, screws: MatNx4x4 = None) -> Mat6xN:
    """
    Args:
        g (Mat4x4): SE(3) element representing end-effector, or any other frame whose velocity is required
        Js (Mat6xN, optional): Spatial Jacobian
        gs (MatNx4x4, optional): SE(3) configuration of each joints
        screws (MatNx4x4, optional): screw axis for joints

    Returns:
        Mat6xN: Body Jacobian
    """
    if Js is None:
        Js = get_spatial_jacobian(gs, screws)
    
    Ad_invs = []
    for j in Js.T:
        Ad_invs.append(se3_derepr(Ad_inv(g, se3_repr(j))))
    
    Jb = np.vstack(Ad_invs).T
    return Jb

def get_world_jacobian(g: Mat4x4, Js: Mat6xN = None, gs: MatNx4x4 = None, screws: MatNx4x4 = None) -> Mat6xN:
    """
    Args:
        g (Mat4x4): SE(3) element representing end-effector, or any other frame whose velocity is required
        Js (Mat6xN, optional): Spatial Jacobian
        gs (MatNx4x4, optional): SE(3) configuration of each joints
        screws (MatNx4x4, optional): screw axis for joints

    Returns:
        Mat6xN: World Jacobian
    """
    if Js is None:
        Js = get_spatial_jacobian(gs, screws)
    
    TeRgs = []
    for j in Js.T:
        TeRgs.append(se3_derepr(right_lifted_action(g, se3_repr(j))))
    
    Jw = np.vstack(TeRgs).T
    return Jw