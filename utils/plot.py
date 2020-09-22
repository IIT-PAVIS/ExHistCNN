import torch
import open3d
import numpy as np
import matplotlib.pyplot as plt
from utils.math import norm

def figure(h=None):
    """
    Open a new figure or select an existing one.

    Parameters
    ----------
    h : integer
        Number of the figure
    """
    plt.figure(h)


def subplot(opt):
    """
    Specify the drawing area.

    Parameters
    ----------
    opt : integer
        A 3-digits integer XYZ where X is the number of rows, Y is the number of columns and Z the row-major index of the XY subplot table
    """
    plt.subplot(opt)


def close(h=None):
    """
    Close a figure.

    Parameters
    ----------
    h : integer
        Number of the figure
    """
    plt.close(h)


def clf():
    """
    Clear the current figure.
    """
    plt.clf()
    plt.show(block=False)


def pause(seconds=0):
    """
    Stop the process.

    Parameters
    ----------
    seconds : float
        Time in seconds to wait; if 0 (default) the user has to close the current figure in order to continue.
    """
    try:
        plt.pause(seconds)
    except:
        # can't invoke "update" command: application has been destroyed
        pass


def grid(enable):
    """
    Eanble/disable the grid on the current figure.

    Parameters
    ----------
    enable : integer, boolean or string
        Could be True, False, 0, 1, "true", "false", "on", "off" (case insensitive)
    """
    if isinstance(enable, str):
        enable = enable.lower()
        if enable in ["on", "true"]:
            enable = True
        if enable in ["off", "false"]:
            enable = False
    plt.grid(enable)
    plt.show(block=False)


def axis(opt):
    """
    Set the axes of the current figure.

    Parameters
    ----------
    opt : list or string
        If list, the format is [x0 x1] or [x0 x1 y0 y1]; if string, should be "equal" or "image"
    """
    if isinstance(opt, str) and opt.lower() == "equal":
        plt.gcf().gca().set_aspect("equal")
    else:
        plt.axis(opt)
    plt.show(block=False)


def xlabel(name):
    """
    Set the label of the X axis of the current figure.

    Parameters
    ----------
    name : string
        Name of the axis
    """
    plt.xlabel(name)
    plt.show(block=False)


def ylabel(name):
    """
    Set the label of the Y axis of the current figure.

    Parameters
    ----------
    name : string
        Name of the axis
    """
    plt.ylabel(name)
    plt.show(block=False)


def title(name):
    """
    Add a title to the current figure.

    Parameters
    ----------
    name : string
        Name of the axis
    """
    if len(plt.gcf().get_axes()) == 0:
        plt.suptitle(name)
    else:
        plt.title(name)
    plt.show(block=False)


def legend():
    """
    Show the legend according to the labels plots.
    """
    try:
        plt.legend()
        plt.show(block=False)
    except:
        pass


def o3d_coordinate(pose, size_axis = 0.2):
    mesh_frame = open3d.create_mesh_coordinate_frame(size=size_axis, origin=[0, 0, 0])
    mesh_frame.transform(pose)
    return mesh_frame


def o3d_trace_rays(cam_pose,  max_range = 5.5, fov_h_angle = 0.34 * np.pi, fov_v_angle = 0.25 * np.pi, angle_gap = 0.01 * np.pi, color = [0.5, 1, 0.5]):
    ray_set = open3d.LineSet()
    cam_forward = np.array([0,0,1])
    cam_left = np.array([-1,0,0])
    cam_up = np.array([0,-1,0])
    points = []
    lines = []
    points.append(cam_pose[:3,3].tolist())
    # prepare the list of view angles
    fov_h_angle_list = np.arange(start=fov_h_angle/2, stop=-fov_h_angle/2, step=-angle_gap)
    fov_h_angle_list = np.append(fov_h_angle_list, -fov_h_angle/2) # ensure the last value is included
    fov_v_angle_list = np.arange(start=fov_v_angle/2, stop=-fov_v_angle/2, step=-angle_gap)
    fov_v_angle_list = np.append(fov_v_angle_list, -fov_v_angle/2) # ensure the last value is included

    for angle_v in fov_v_angle_list:
        up_offset = cam_up * np.tan(angle_v)
        for angle_h in fov_h_angle_list:
            left_offset = cam_left * np.tan(angle_h)
            direction_unit = cam_forward + up_offset + left_offset
            direction_unit = direction_unit / np.linalg.norm(direction_unit)
            direction_unit = np.dot(cam_pose[:3,:3], direction_unit)
            point = cam_pose[:3,3] + direction_unit*max_range
            points.append(point.tolist())

    for i in range(len(points)-1):
        line = [0, i+1]
        lines.append(line)

    colors = [color for i in range(len(lines))]
    # get vector camera_look_at
    # set ray_set
    ray_set.points = open3d.Vector3dVector(points)
    ray_set.lines = open3d.Vector2iVector(lines)
    ray_set.colors = open3d.Vector3dVector(colors)

    return ray_set


def imshow(image):
    """
    Show an image (non blocking).
    Image could be either a numpy array or a pytorch tensor.

    Parameters
    ----------
    image : numpy.ndarray (HW[C] format) or torch.Tensor ([C]HW format)
        Matrix or tensor to visualize
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if len(image.shape) == 3:
            image = np.swapaxes(image, 0, 1)
            image = np.swapaxes(image, 1, 2)

    if len(image.shape) == 3:
        if image.shape[-1] not in [1, 3]:
            raise Exception("Invalid image format. Only single or three channels images are supported")

    plt.imshow(image)
    plt.show(block=False)


def imagesc(image):
    """
    Show an image after normalization (non blocking).
    Image could be either a numpy array or a pytorch tensor.

    Parameters
    ----------
    image : numpy.ndarray (HW[C] format) or torch.Tensor ([C]HW format)
        Matrix or tensor to visualize
    """

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        if len(image.shape) == 3:
            image = np.swapaxes(image, 0, 1)
            image = np.swapaxes(image, 1, 2)

    src = image.copy()
    dst = src - np.min(src)
    maxval = np.max(dst)
    if maxval > 0:
        dst = dst / maxval
    imshow(dst)


def plot(X, Y=None, clr="b", label="", lwidth=1.5):
    """
    2D plot using matplotlib.

    Parameters
    ----------
    X : numpy.ndarray
        Data array; if Y is None, shape of X must be Nx1 or Nx2
    Y : numpy.ndarray (default: None)
        Second data array
    clr : string, float or 3d array (default: "b")
        Color of the plot
    label : string, (default: "")
        Legend string of the plot
    """

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        X = np.squeeze(X)

    if Y is None:
        if len(X.shape) == 1:
            xx = np.arange(X.size)
            yy = X
        if len(X.shape) == 2:
            xx = X[:,0]
            yy = X[:,1]
    else:
        xx = X
        yy = Y

    plt.plot(xx, yy, clr, label=label, linewidth=lwidth)
    plt.show(block=False)


def plot3(X, Y=None, Z=None, clr="b", xlabel=None, ylabel=None):
    """
    3D plot using matplotlib.

    Parameters
    ----------
    X : numpy.ndarray
        Data array; if Y and Z are None, shape of X must be Nx3
    Y : numpy.ndarray (default: None)
        Second data array
    Z : numpy.ndarray (default: None)
        Third data array
    clr : string, float or 3d array (default: "b")
        Color of the plot
    xlabel : string, (default: None)
        Legend string of the x-axis
    ylabel : string, (default: None)
        Legend string of the y-axis
    """

    # TODO
    pass


def scatter(X, Y=None, clr="b", mrk="o", s=30):
    """
    2D scatter plot using matplotlib.

    Parameters
    ----------
    X : numpy.ndarray
        Data array; if Y is None, shape of X must be Nx2
    Y : numpy.ndarray (default: None)
        Second data array
    clr : string, float or 3d array (default: "b")
        Color of the plot
    """

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)
        X = np.squeeze(X)

    if Y is None:
        if len(X.shape) == 1:
            xx = np.arange(X.size)
            yy = X
        if len(X.shape) == 2:
            xx = X[:,0]
            yy = X[:,1]
    else:
        xx = X
        yy = Y
    plt.scatter(xx, yy, c=clr, marker=mrk, s=s)
    plt.show(block=False)


def scatter3(X, Y=None, Z=None, clr=None, backend="open3d"):
    """
    3D scatter plot using Open3D lib.

    Parameters
    ----------
    X : numpy.ndarray
        Data array; if Y and Z are None, shape of X must be Nx3
    Y : numpy.ndarray (default: None)
        Second data array
    Z : numpy.ndarray (default: None)
        Third data array
    clr : string, float or 3d array (default: None)
        Color of the plot
    backend : string (default: "open3d")
        Engine to render the 3d plot; must be "open3d" or "matplotlib"
    """

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    if Y is None and Z is None:
        xyz = X
    else:
        xyz = np.stack((X,Y,Z),axis=-1)

    # if xyz.shape[0] > 1000:
    #     backend = "open3d"

    if backend == "open3d":
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(xyz)
        if clr is not None:
            pcd.colors = open3d.Vector3dVector(clr)
        open3d.draw_geometries([pcd])
    elif backend == "matplotlib":
        fig = plt.gcf()
        ax = fig.gca(projection="3d")
        # axis("equal")
        ax.scatter3D(xyz[:,0],xyz[:,1],xyz[:,2], color=clr)
        plt.show(block=False)
    else:
        raise NotImplementedError()


def arrow3(location, target, length=None, clr="b"):
    """
    Draw a 3D arrow using quiver.

    Parameters
    ----------
    location : numpy.ndarray
        Starting point of the arrow
    target : numpy.ndarray
        Target point of the arrow
    length : float (default: None)
        Length of the arrow
    clr : string, float or 3d array (default: "b")
        Color of the plot
    """

    fig = plt.gcf()
    ax = fig.gca(projection="3d")
    ax.set_aspect("equal")

    if not isinstance(location, np.ndarray):
        location = np.asarray(location)
    if not isinstance(target, np.ndarray):
        target = np.asarray(target)

    direction = target - location
    L = norm(direction)

    direction = direction / L

    if length is not None:
        L = length

    x, y, z = location
    u, v, w = direction

    ax.quiver(x, y, z, u, v, w, length=L, color=clr)

    plt.show(block=False)


def cartesiansystem(posemtx, length=1.0):
    """
    Draw the 3D cartesian axis system according to pose matrix.

    Parameters
    ----------
    posemtx : numpy.ndarray
        Pose matrix of the reference system to plot as a set of 3d arrows
    length : float (default: 1.0)
        Length of the arrows
    """

    if not isinstance(posemtx, np.ndarray):
        posemtx = np.asarray(posemtx)

    S0 = np.asarray([[0,0,0,1],[0,0,0,1],[0,0,0,1]], dtype=np.float32).T
    S0 = np.matmul(posemtx, S0)
    S1 = np.asarray([[1,0,0,1],[0,1,0,1],[0,0,1,1]], dtype=np.float32).T
    S1 = np.matmul(posemtx, S1)

    arrow3(location=S0[:-1,0], target=S1[:-1,0], length=length, clr="r")
    arrow3(location=S0[:-1,1], target=S1[:-1,1], length=length, clr="g")
    arrow3(location=S0[:-1,2], target=S1[:-1,2], length=length, clr="b")

    plt.show(block=False)
