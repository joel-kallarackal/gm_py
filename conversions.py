import numpy as np
from typing import Tuple

def SE2_from_xytheta(x: float, y: float, theta: float) -> np.ndarray:
    """
    Convert (x, y, theta) to SE(2) matrix.
    
    Parameters:
    -----------
    x : float
        x-coordinate translation
    y : float
        y-coordinate translation
    theta : float
        rotation angle in radians
    
    Returns:
    --------
    T : np.ndarray
        3x3 SE(2) matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    
    T = np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ])
    return T


def xytheta_from_SE2(T: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert SE(2) matrix to (x, y, theta).
    
    Parameters:
    -----------
    T : np.ndarray
        3x3 SE(2) matrix
    
    Returns:
    --------
    x : float
        x-coordinate translation
    y : float
        y-coordinate translation
    theta : float
        rotation angle in radians
    """
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    
    return x, y, theta


def SE3_from_xyz_rpy(x: float, y: float, z: float, 
                      roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert position (x, y, z) and Roll-Pitch-Yaw angles to SE(3) matrix.
    Uses ZYX Euler angle convention (intrinsic rotations).
    
    Parameters:
    -----------
    x, y, z : float
        Position coordinates
    roll : float
        Rotation around x-axis (radians)
    pitch : float
        Rotation around y-axis (radians)
    yaw : float
        Rotation around z-axis (radians)
    
    Returns:
    --------
    T : np.ndarray
        4x4 SE(3) matrix
    """
    # Rotation around x-axis (roll)
    cr = np.cos(roll)
    sr = np.sin(roll)
    Rx = np.array([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])
    
    # Rotation around y-axis (pitch)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    Ry = np.array([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])
    
    # Rotation around z-axis (yaw)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    Rz = np.array([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])
    
    # Combined rotation: R = Rz @ Ry @ Rx (ZYX convention)
    R = Rz @ Ry @ Rx
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def xyz_rpy_from_SE3(T: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Convert SE(3) matrix to position (x, y, z) and Roll-Pitch-Yaw angles.
    Uses ZYX Euler angle convention.
    
    Parameters:
    -----------
    T : np.ndarray
        4x4 SE(3) matrix
    
    Returns:
    --------
    x, y, z : float
        Position coordinates
    roll, pitch, yaw : float
        Euler angles in radians
    """
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    R = T[:3, :3]
    
    # Extract Euler angles from rotation matrix (ZYX convention)
    # Handle gimbal lock at pitch = ±90 degrees
    if abs(R[2, 0]) < 1 - 1e-6:
        pitch = np.arcsin(-R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock case
        yaw = 0  # Set yaw to zero by convention
        if R[2, 0] < 0:  # pitch = pi/2
            pitch = np.pi / 2
            roll = yaw + np.arctan2(R[0, 1], R[0, 2])
        else:  # pitch = -pi/2
            pitch = -np.pi / 2
            roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])
    
    return x, y, z, roll, pitch, yaw


def SE3_from_xyz_axisangle(x: float, y: float, z: float,
                            axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Convert position and axis-angle to SE(3) matrix.
    
    Parameters:
    -----------
    x, y, z : float
        Position coordinates
    axis : np.ndarray
        3D rotation axis (will be normalized)
    angle : float
        Rotation angle in radians
    
    Returns:
    --------
    T : np.ndarray
        4x4 SE(3) matrix
    """
    # Normalize axis
    axis = axis / np.linalg.norm(axis)
    
    # Rodrigues' formula for rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def xyz_axisangle_from_SE3(T: np.ndarray) -> Tuple[float, float, float, np.ndarray, float]:
    """
    Convert SE(3) matrix to position and axis-angle.
    
    Parameters:
    -----------
    T : np.ndarray
        4x4 SE(3) matrix
    
    Returns:
    --------
    x, y, z : float
        Position coordinates
    axis : np.ndarray
        3D rotation axis (unit vector)
    angle : float
        Rotation angle in radians
    """
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    R = T[:3, :3]
    
    # Compute rotation angle
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    
    if abs(angle) < 1e-9:
        # Near identity - arbitrary axis
        axis = np.array([0, 0, 1])
    elif abs(angle - np.pi) < 1e-6:
        # Near pi rotation
        B = R + np.eye(3)
        norms = np.linalg.norm(B, axis=0)
        k = np.argmax(norms)
        axis = B[:, k] / np.linalg.norm(B[:, k])
    else:
        # General case
        axis = (1 / (2 * np.sin(angle))) * np.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ])
    
    return x, y, z, axis, angle


def SE3_from_xyz_quaternion(x: float, y: float, z: float,
                             qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """
    Convert position and quaternion to SE(3) matrix.
    Quaternion format: q = qw + qx*i + qy*j + qz*k
    
    Parameters:
    -----------
    x, y, z : float
        Position coordinates
    qw, qx, qy, qz : float
        Quaternion components (will be normalized)
    
    Returns:
    --------
    T : np.ndarray
        4x4 SE(3) matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Convert quaternion to rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T


def xyz_quaternion_from_SE3(T: np.ndarray) -> Tuple[float, float, float, float, float, float, float]:
    """
    Convert SE(3) matrix to position and quaternion.
    
    Parameters:
    -----------
    T : np.ndarray
        4x4 SE(3) matrix
    
    Returns:
    --------
    x, y, z : float
        Position coordinates
    qw, qx, qy, qz : float
        Quaternion components
    """
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]
    
    R = T[:3, :3]
    
    # Convert rotation matrix to quaternion
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return x, y, z, qw, qx, qy, qz
