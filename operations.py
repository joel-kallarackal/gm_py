import numpy as np

# Type aliases (shape conventions only)
Vec3 = np.ndarray       # (3,)
Vec6 = np.ndarray       # (6,)
Mat3x3 = np.ndarray     # (3,3)
Mat4x4 = np.ndarray     # (4,4)
MatrixNxN = np.ndarray  # square matrix

def se2_repr(v: Vec3) -> Mat3x3:
    '''
        Return the matrix representation of a vector in se(2).
        Input: v = [vx, vy, w] where vx, vy is the translation part and w is the rotation part.
        Output: X = [[0, -w, vx],
                     [w,  0, vy],
                     [0,  0,  0]]
    '''
    vx, vy, w = v
    return np.array([[0.0, -w, vx], [w, 0.0, vy], [0.0, 0.0, 0.0]], dtype=float)

def se2_derepr(X: Mat3x3) -> Vec3:
    '''
        Return the vector representation of a matrix in se(2).
        Input: X = [[0, -w, vx],
                    [w,  0, vy],
                    [0,  0,  0]]
        Output: v = [vx, vy, w] where vx, vy is the translation part and w is the rotation part.
            '''
    X = np.array(X)
    return np.array([X[0, 2], X[1, 2], X[1, 0]], dtype=float)

def se3_repr(v: Vec6) -> Mat4x4:
    """
    Convert 6-vector to se(3) 4x4 matrix.
    Input: v = [vx, vy, vz, wx, wy, wz]
    Output: X = 4x4 se(3) matrix
    """
    vx, vy, vz, wx, wy, wz = v
    X = np.array([
        [0.0, -wz,  wy, vx],
        [wz,  0.0, -wx, vy],
        [-wy, wx,  0.0, vz],
        [0.0, 0.0, 0.0, 0.0]
    ], dtype=float)
    return X

def se3_derepr(X: Mat4x4) -> Vec6:
    """
    Convert se(3) 4x4 matrix to 6-vector.
    Output: v = [vx, vy, vz, wx, wy, wz]
    """
    vx, vy, vz = X[0, 3], X[1, 3], X[2, 3]
    wx, wy, wz = X[2,1], X[0,2], X[1,0]
    return np.array([vx, vy, vz, wx, wy, wz], dtype=float)



def left_lifted_action(g: MatrixNxN, X: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); X is the se(2) algebra element (matrix).
    
    Maps body velocity X(right groupwise velocity) to world velocity at g
    """
    return g @ X

def inv_left_lifted_action(g: MatrixNxN, g_dot: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); g_dot is at the tangent space at g (matrix).
    
    Maps world velocity at g to body velocity(right groupwise velocity)
    """
    return np.linalg.inv(g) @ g_dot


def right_lifted_action(g: MatrixNxN, X: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); X is the se(2) algebra element (matrix).
    
    Maps spatial velocity X(left groupwise velocity) to world velocity at g
    """
    return X @ g

def inv_right_lifted_action(g: MatrixNxN, g_dot: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); g_dot is at the tangent space at g (matrix).
    
    Maps world velocity at g to spatial velocity(left groupwise velocity)
    """
    return g_dot @ np.linalg.inv(g) 


def Ad(g: MatrixNxN, X: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); X is the se(2) algebra element (matrix).
    
    Adjoint: Converts Body Velocity(right groupwise velocity) to Spatial Velocity(left groupwise velocity)
    """
    return g @ X @ np.linalg.inv(g)

def Ad_inv(g: MatrixNxN, X: MatrixNxN) -> MatrixNxN:
    """
    g is the SE(2) group element (matrix); X is the se(2) algebra element (matrix).
    
    Adjoint Inverse: Converts Spatial Velocity(left groupwise velocity) to Body Velocity(right groupwise velocity)
    """
    return np.linalg.inv(g) @ X @ g


def ad(X: MatrixNxN, Y: MatrixNxN) -> MatrixNxN:
    """
    X and Y are se(2) algebra elements (matrices).
    Computes Lie Bracket / Infinitesimal Adjoint
    """
    return X @ Y - Y @ X



def se2_exp(X: Mat3x3) -> Mat3x3:
    """
    Exponential map from se(2) to SE(2)
    """
    omega = X[1, 0]
    vx = X[0, 2]
    vy = X[1, 2]
    v = np.array([vx, vy])

    if np.abs(omega) < 1e-9:
        # Pure translation
        R = np.eye(2)
        t = v
    else:
        # Rotation matrix
        c = np.cos(omega)
        s = np.sin(omega)
        R = np.array([[c, -s],
                      [s,  c]])

        # V matrix
        V = (1.0 / omega) * np.array([
            [s, -(1 - c)],
            [(1 - c), s]
        ])

        t = V @ v

    g = np.eye(3)
    g[:2, :2] = R
    g[:2, 2] = t
    return g

def SE2_log(g: Mat3x3) -> Mat3x3:
    """
    Logarithm map from SE(2) to se(2)
    """
    # Extract rotation matrix and translation vector
    R = g[:2, :2]
    p = g[:2, 2]
    
    # Compute rotation angle theta from the rotation matrix
    # R = [[cos(theta), -sin(theta)],
    #      [sin(theta),  cos(theta)]]
    theta = np.arctan2(R[1, 0], R[0, 0])
    
    if np.abs(theta) < 1e-10:
        # Near identity, use first-order approximation
        V_inv = np.eye(2) - 0.5 * np.array([[0, -theta], [theta, 0]])
    else:
        # General case
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / theta
        V_inv = (1 / (A**2 + B**2)) * np.array([
            [A, B],
            [-B, A]
        ])
    
    rho = V_inv @ p
    
    X = np.array([
        [0, -theta, rho[0]],
        [theta, 0, rho[1]],
        [0, 0, 0]
    ])
    
    return X

def se3_exp(X : Mat4x4) -> Mat4x4:
    """
    Exponential map from se(3) to SE(3)
    """
    W = X[:3,:3]
    v = X[:3,3]

    w = np.array([W[2,1], W[0,2], W[1,0]], dtype=float)
    theta = np.linalg.norm(w)

    if theta < 1e-9:
        R = np.eye(3)
        t = v
    else:
        # Rodrigues formula
        W_norm = W / theta
        R = np.eye(3) + np.sin(theta) * W_norm + (1 - np.cos(theta)) * (W_norm @ W_norm)
        # V matrix
        V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * W + (theta - np.sin(theta)) / (theta**3) * (W @ W)
        t = V @ v

    g = np.eye(4)
    g[:3,:3] = R
    g[:3,3] = t
    return g

def SE3_log(g: Mat4x4) -> Mat4x4:
    """
    Logarithm map from SE(3) to se(3)
    """
    R = g[:3, :3]
    t = g[:3, 3]
    
    # Compute rotation angle
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical stability
    theta = np.arccos(cos_theta)
    
    if abs(theta) < 1e-9:
        # Near identity - use first order approximation
        W = 0.5 * (R - R.T)
        # V ≈ I for small theta
        v = t
        
    elif abs(theta - np.pi) < 1e-6:
        # Near pi rotation - special case to avoid singularity
        # Find the eigenvector corresponding to eigenvalue 1 (rotation axis)
        # This is equivalent to finding the column of (R + I) with largest norm
        B = R + np.eye(3)
        # Find column with largest norm
        norms = np.linalg.norm(B, axis=0)
        k = np.argmax(norms)
        omega = B[:, k] / np.linalg.norm(B[:, k])
        omega = theta * omega
        
        # Skew-symmetric matrix
        W = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        # V matrix for theta ≈ pi
        V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * W + (theta - np.sin(theta)) / (theta**3) * (W @ W)
        v = np.linalg.solve(V, t)
        
    else:
        # General case
        W = (theta / (2 * np.sin(theta))) * (R - R.T)
        
        # V matrix
        V = np.eye(3) + (1 - np.cos(theta)) / (theta**2) * W + (theta - np.sin(theta)) / (theta**3) * (W @ W)
        v = np.linalg.solve(V, t)
    
    X = np.zeros((4, 4))
    X[:3, :3] = W
    X[:3, 3] = v
    return X
