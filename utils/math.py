import open3d
import numpy as np


ZERO = 1E-6


def flip(arr, axis):
    """
    Flip a numpy array.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    axis : integer
        Axis along which the array is flipped

    Returns
    -------
    arr : numpy.ndarray
        Flipped version of the input array
    """

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "horizontal":
            axis = 0
        if axis == "vertical":
            axis = 1

    arr = np.flip(arr, axis)
    return arr


def normalize(arr, axis=None):
    """
    Normalize a vector between 0 and 1.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    axis : integer
        Axis along which normalization is computed

    Returns
    -------
    arr : numpy.ndarray
        Normalized version of the input array
    """

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    arr = arr - np.min(arr, axis)
    M = np.max(arr, axis)
    if np.sum(np.abs(M)) > 0:
        arr = arr / M

    return arr


def norm(arr, axis=None):
    """
    Compute the norm of a vector.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array
    axis : integer
        Axis along which norm is computed

    Returns
    -------
        Norm of the input array
    """

    return np.sqrt(np.sum(arr**2, axis=axis))


def lookat(location, target, dtype=np.float32):
    """
    Compute the pose matrix (rotation, translation)
    of the camera which position is "location" and
    looks at "target" coordinates.
    X axis is parallel to the XY (world) plane
    Z axis points to "target".
    Y is the cross product of Z and X so that it point
    to the positive z-hemispace

    Parameters
    ----------
    location : numpy.ndarray
        Array with 3 values
    target : numpy.ndarray
        Array with 3 values
    dtype : type (default: numpy.float32)
        Desired output data type (e.g. np.float32)

    Returns
    -------
    posemtx : numpy.ndarray
        4x4 pose matrix
    """

    world_z = np.asarray([0, 0, 1])

    cam_z = target - location
    cam_z = cam_z / (norm(cam_z) + ZERO)

    cam_x = np.cross(cam_z, world_z)
    cam_x = cam_x / (norm(cam_x) + ZERO)

    cam_y = np.cross(cam_z, cam_x)
    cam_y = cam_y / (norm(cam_y) + ZERO)

    R = np.stack((cam_x, cam_y, cam_z)).T
    t = location

    posemtx = np.zeros((4,4))
    posemtx[0:3,0:3] = R
    posemtx[0:3, 3] = t
    posemtx[3,3] = 1

    return posemtx.astype(dtype)


def downsample(X, factor):
    """
    Downsample data matrix rows.

    Parameters
    ----------
    X : numpy.ndarray
        Array to downsample
    factor : integer
        Sampling factor

    Returns
    -------
    pts : numpy.ndarray
        Extracted points
    """

    idx = np.arange(0, X.shape[0], factor)
    pts = X[idx]

    return pts


def sampleprimitivesurface(surf="sphere", n=100):
    """
    Sample the surface of a primitive shape.

    Parameters
    ----------
    surf : string (default: "sphere")
        Type of the surface to sample. Currently supported:
        - sphere
    n : integer (default: 100)
        Number of samples

    Returns
    -------
    pts : numpy.ndarray
        Array of positions with shape nx3
    """

    pts = None

    if surf == "sphere":
        pts = _sample_sphere(n)
    else:
        raise NotImplementedError()

    return pts


def _sample_sphere(n):
    """
    Distributing many points on a sphere.
    SAFF, Edward B.; KUIJLAARS, Amo BJ.
    The mathematical intelligencer, 1997, 19.1: 5-11.

    Parameters
    ----------
    n : integer
        Number of samples

    Returns
    -------
    pts : numpy.ndarray
        Array of positions with shape nx3
    """

    if n < 1:
        return None

    s = 3.6 / np.sqrt(n)
    dz = 2.0 / n
    angle = 0.0
    z = 1 - dz / 2

    pts = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        r = np.sqrt(1 - z * z)
        # compute coordinates
        x = np.cos(angle) * r
        y = np.sin(angle) * r
        pts[i] = [x, y, z]
        # update
        angle = angle + s / r
        z = z - dz

    return pts


class NearestNeighbors():


    def __init__(self, xyz):
        self.pcd = open3d.PointCloud()
        self.pcd.points = open3d.Vector3dVector(xyz)
        self.pcd_tree = open3d.KDTreeFlann(self.pcd)


    def search_knn(self, p, k=1):
        if isinstance(p, int):
            k = k + 1
            point = self.pcd.points[p]
        else:
            point = p
        [k, idx, _] = self.pcd_tree.search_knn_vector_3d(point, k)
        idx = np.asarray(idx).reshape(-1,)
        if idx.size > 0 and isinstance(p, int):
            idx = idx[1:]
        return idx


    def search_radius(self, p, r=1):
        point = self.pcd.points[p] if isinstance(p, int) else p
        [k, idx, _] = self.pcd_tree.search_radius_vector_3d(point, r)
        idx = np.asarray(idx).reshape(-1,)
        if idx.size > 0 and isinstance(p, int):
            idx = idx[1:]
        return idx

class Gaussian():
    def __init__(self,mu_,sig_):
        self.mu = mu_
        self.sig = sig_
        
    def computeY(self,x):
        return np.exp(-np.power(x - self.mu, 2.) / (2 * np.power(self.sig, 2.)))
        
def knearestneighbors(xyz, i, k=None):
    """
    Compute the k nearest neighbors algorithm on the XYZ point cloud in
    the format Nx3.
    "i" is the index of the point (of the point cloud) used as reference.

    Parameters
    ----------
    xyz : numpy.ndarray
        Data array with shape Nx3
    i : integer
        Index of the reference point in xyz
    k : integer
        Number of neighbors to extract

    Returns
    -------
    idx : numpy.ndarray
        Array of indices of the neighbors
    """
    print("File \"" + __file__ + "\"")
    print("  WARNING: use \"NearestNeighbors\" class for a more efficient interface")

    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(xyz)

    pcd_tree = open3d.KDTreeFlann(pcd)

    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k)

    return idx


def rotation_to_euler(R, mode="XYZ"):
    if mode == "XYZ":
        theta1 = np.math.atan2(R[1,2], R[2,2])
        c2 = np.sqrt(R[0,0]**2 + R[0,1]**2)
        theta2 = np.math.atan2(-R[0,2], c2)
        s1 = np.math.sin(theta1)
        c1 = np.math.cos(theta1)
        theta3 = np.math.atan2(s1*R[2,0] - c1*R[1,0], c1*R[1,1] - s1*R[2,1])
    else:
        raise NotImplementedError()
    return [theta1, theta2, theta3]


def rmtx_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[1,1] = c
    mtx[1,2] = s
    mtx[2,1] = -s
    mtx[2,2] = c
    return mtx


def rmtx_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[0,0] = c
    mtx[0,2] = -s
    mtx[2,0] = s
    mtx[2,2] = c
    return mtx


def rmtx_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    mtx = np.eye(3, dtype=np.float32)
    mtx[0,0] = c
    mtx[0,1] = s
    mtx[1,0] = -s
    mtx[1,1] = c
    return mtx


def euler_to_rotation(theta1, theta2, theta3, mode="XYZ"):
    if mode == "XYZ":
        Rx = rmtx_x(theta1)
        Ry = rmtx_y(theta2)
        Rz = rmtx_z(theta3)
        R = np.matmul(Rx, np.matmul(Ry, Rz))
    else:
        raise NotImplementedError()
    return R


def get_max_ind(array_in, option="first"):
    largest_val = np.max(array_in)
    largest_ind_list = np.nonzero(array_in==largest_val)[0]
    if largest_ind_list.shape[0]>1:
        if option == "first":
            print("WARN: Multiple maximum occurs, use the first one")
            largest_ind = largest_ind_list[0]
        elif option == "random":
            print("WARN: Multiple maximum occurs, use a random one")
            largest_ind = np.random.choice( largest_ind_list)
    else:
        largest_ind = largest_ind_list[0]

    return largest_ind


def quaternion_to_rotation(q):
    w, x, y, z = q

    r11 = w*w + x*x - y*y - z*z
    r12 = 2*(x*y - w*z)
    r13 = 2*(w*y + x*z)

    r21 = 2*(x*y + w*z)
    r22 = w*w - x*x + y*y - z*z
    r23 = 2*(y*z - w*x)

    r31 = 2*(x*z - w*y)
    r32 = 2*(y*z + w*x)
    r33 = w*w - x*x - y*y + z*z

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33],
    ])


# to compute the theoretical number of combinations
# @Input:the total number of elements
#        the number of elements to be selected
#
# @Output: the number of combination numbers
def get_comb_number(n_total, n_sel):
    return np.math.factorial(n_total)/(np.math.factorial(n_sel)*np.math.factorial(n_total-n_sel))


# to compute a rotated vector with a rotation matrix
# @Input:the rotation matrix
#        vector
#
# @Output: rotated vector
def rotate_unit_vec(rotM, unit_vec):
    rotm_x = rotM[0, :3]
    rotm_y = rotM[1, :3]
    rotm_z = rotM[2, :3]

    x_ = np.dot(rotm_x, unit_vec)
    y_ = np.dot(rotm_y, unit_vec)
    z_ = np.dot(rotm_z, unit_vec)

    return np.array([x_, y_, z_])
