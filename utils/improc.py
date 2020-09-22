import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.signal import convolve2d
from skimage import transform

def imread(filename, size=None, dtype=None, fmt=None):
    """
    Load an image file from disk.

    Parameters
    ----------
    filename : string
        Path of the image file
    size : integer (default: None)
        Desired resolution as (W,H) tuple
    dtype : type (default: None)
        Desired data type (e.g. np.float32)
    fmt : string (default: None)
        Format of the image (e.g. "F" or "RGBA")

    Returns
    -------
    image : numpy.ndarray
        Loaded image
    """
    image = Image.open(filename)
    if size is not None and len(size) == 2:
        image = image.resize(size)
    if fmt is not None:
        image = image.convert(fmt)
    image = np.asarray(image)
    if dtype is not None:
        image = image.astype(dtype)
    return image


def imwrite(filename, image, fmt=None):
    """
    Save an image file from disk.

    Parameters
    ----------
    filename : string
        Path of the image file
    image : numpy.ndarray
        Array to save as image
    fmt : string (default: None)
        Format of the image (e.g. "F" or "RGBA")
    """
    src = Image.fromarray(image, fmt)
    src.save(filename)


def imresize(image, size):
    """
    Resize a ndarray.

    Parameters
    ----------
    image : numpy.ndarray
        Array to resize
    size : integer
        Desired resolution as (W,H) tuple

    Returns
    -------
    dst : numpy.ndarray
        Resized image
    """
    dst = cv2.resize(image, size)
    return dst


def imscale(image, scale, **kwargs):
    """
    Scale an ndarray.
    Reference: https://stackoverflow.com/a/37121993/1358091

    Parameters
    ----------
    image : numpy.ndarray
        Array to scale
    scale : integer
        Scale factor. Must be positive.

    Returns
    -------
    out : numpy.ndarray
        Scaled image
    """

    if scale <= 0:
        raise ValueError()

    h, w = image.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (scale,) * 2 + (1,) * (image.ndim - 2)

    # Zooming out
    if scale < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * scale))
        zw = int(np.round(w * scale))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(image)
        out[top:top+zh, left:left+zw] = zoom(image, zoom_tuple, **kwargs)

    # Zooming in
    elif scale > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / scale))
        zw = int(np.round(w / scale))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(image[top:top+zh, left:left+zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `image` due to rounding, so
        # trim off any extra pixels at the edges
        out = imresize(out, (w,h))
        # trim_top = ((out.shape[0] - h) // 2)
        # trim_left = ((out.shape[1] - w) // 2)
        # out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If scale == 1, just return the input array
    else:
        out = image
    return out


def imrotate(image, angle):
    """
    Rotate a ndarray.

    Parameters
    ----------
    image : numpy.ndarray
        Array to rotate
    angle : float
        Rotation angle in degrees

    Returns
    -------
    dst : numpy.ndarray
        Rotated image
    """
    rows, cols = image.shape[:2]
    R = cv2.getRotationMatrix2D((rows//2, cols//2), angle, 1)
    dst = cv2.warpAffine(image, R, (cols, rows))
    return dst


def imshift(image, dx, dy):
    """
    Inplace translation of the input image.

    Parameters
    ----------
    image : numpy.ndarray
        Array to shift
    dx : float
        Offset along x axis
    dy : float
        Offset along y axis

    Returns
    -------
    dst : numpy.ndarray
        Shifted image
    """
    rows, cols = image.shape[:2]
    T = np.asarray([[1,0,dx],[0,1,dy]], dtype=np.float32)
    dst = cv2.warpAffine(image, T, (cols, rows))
    return dst


def imblend(image1, image2, alpha=0.5):
    """
    Blend two images.
    If one is RGB and the other is BW, the single channel image
    is converted to RGB.

    Parameters
    ----------
    image1 : numpy.ndarray
        First image in the format HW[3]
    image2 : numpy.ndarray
        Second image in the format HW[3]
    alpha : float (default: 0.5)
        Normalized blending factor or mask

    Returns
    -------
    dst : numpy.ndarray
        Blended image
    """

    if len(image1.shape) == 2 and len(image2.shape) == 3:
        image1 = gray2rgb(image1)
    if len(image1.shape) == 3 and len(image2.shape) == 2:
        image2 = gray2rgb(image2)

    mask = alpha
    if isinstance(alpha, np.ndarray):
        nChannels = image1.shape[-1]
        mask = alpha.copy()
        mask = np.stack([mask]*nChannels, axis=-1)
        mask = mask.astype(np.float32)
    dst = mask * image1 + (1. - mask) * image2
    return dst


def impad(image, margin, value=0):
    """
    Add a constant pad of size 'margin' with value 'value'
    to the input image or ndarray.

    Parameters
    ----------
    image : numpy.ndarray
        Image to pad in the format HW[C]
    margin : integer
        Pad size; must be greater than 0
    value : image.dtype
        Constant pad value; its type must be the same of the input image

    Returns
    -------
    padded_image : numpy.ndarray
        Padded image in the format HW[C]
    """

    assert isinstance(image, np.ndarray)
    assert isinstance(margin, int) and margin > 0
    # assert isinstance(value, image.dtype)

    if len(image.shape) == 2:
        padded_image = np.pad(image, margin, "constant", constant_values=value)
    else:
        nchannels = image.shape[-1]
        padded_image = []
        for c in range(nchannels):
            padded_channel = np.pad(image[:,:,c], margin, "constant", constant_values=value)
            padded_image.append(padded_channel)
        padded_image = chw2hwc(np.asarray(padded_image))
    return padded_image


def imcrop(image, centroid, roisize):
    """
    Crop the input image or ndarray.

    Parameters
    ----------
    image : numpy.ndarray
        Image to crop in the format HW[C]
    centroid : list, tuple or np.array
        Coordinates of the crop center
    roisize : list, tuple or np.array
        Dimensions of the area to crop

    Returns
    -------
    crop_area : numpy.ndarray
        Cropped area from the input image
    r0, r1 : integers
        Row coordinates of the cropped area
    c0, c1 : integers
        Column coordinates of the cropped area
    """

    assert isinstance(image, np.ndarray)
    assert isinstance(centroid, list) or isinstance(centroid, tuple) or isinstance(centroid, np.ndarray)
    assert isinstance(roisize, list) or isinstance(roisize, tuple) or isinstance(roisize, np.ndarray)

    roisize = np.asarray(roisize)

    hroi = roisize // 2
    if roisize.size == 2:
        hh, hw = hroi
    else:
        hh, hw = hroi, hroi

    cu, cv = int(centroid[0]), int(centroid[1])
    imcpy = impad(image, np.maximum(hh,hw))
    r0, r1 = cv + hh, cv + roisize + hh
    c0, c1 = cu + hw, cu + roisize + hw
    print(r0, r1,c0, c1)
    crop = imcpy[r0:r1, c0:c1,:]
    return crop, r0, r1, c0, c1

def crop_border(image_name, height_border_ratio, width_border_ratio):
    img = imread(image_name)
    height, width, _ = img.shape
    border_height = int(height*height_border_ratio/2.0)
    border_width = int(width*width_border_ratio/2.0)

    img = img[border_height:(height - border_height), border_width:(width - border_width)]
    return img
    
def gray2rgb(image):
    """
    Convert grayscale image to rgb.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    dst : numpy.ndarray
        Converted image
    """
    dst = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    dst = dst.astype(image.dtype)
    return dst


def rgb2gray(image):
    """
    Convert rgb image to grayscale.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    dst : numpy.ndarray
        Converted image
    """
    dst = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dst = dst.astype(image.dtype)
    return dst


def hwc2chw(image):
    """
    Change channel positions from HWC format to CHW.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    dst : numpy.ndarray
        Converted image
    """
    dst = np.swapaxes(image,1,2)
    dst = np.swapaxes(dst,0,1)
    return dst


def chw2hwc(image):
    """
    Change channel positions from CHW format to HWC.

    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    dst : numpy.ndarray
        Converted image
    """
    dst = np.swapaxes(image,0,1)
    dst = np.swapaxes(dst,1,2)
    return dst


def repmat(arr, n, axis=0):
    """
    Repeat an array n times.

    Parameters
    ----------
    arr : numpy.ndarray or list
        Array to replicate
    n : integer
        Number of repetitions
    axis : integer
        Axis along which the repetition is performed

    Returns
    -------
    dst : numpy.ndarray
        Repeated array
    """
    dst = np.stack((arr,)*n, axis=axis)
    dst = dst.astype(arr.dtype)
    return dst



def denoise(depth, kernel_size = 3):
    window = np.ones((kernel_size, kernel_size))
    window /= np.sum(window)
    denoised_depth = convolve2d(depth,window, mode = "same", boundary = "symm")
    return denoised_depth


def normalise_gray_image(image_gray, old_limit, new_limit):
    max_value = min(image_gray.reshape(-1).max(), old_limit)
    image_gray[image_gray > max_value] = max_value
    # normalise value into 255
    if max_value == 0:
        print("WARN:: max_depth is 0, the depth image is going to be normalised!")
        image_gray = image_gray
    else:
        image_gray = image_gray / float(max_value) * new_limit
    return image_gray, max_value


# this function will shrink the image, with the option with zero padding so that the dimension of the
def imshrink(input_image, half_large_fov, half_small_fov, padding = False):
    # get the shape of the input image
    height, width = input_image.shape[:2]
    if padding:
        new_width = int(np.tan(half_small_fov[0])/np.tan(half_large_fov[0])*width)
        new_height = int(np.tan(half_small_fov[1])/np.tan(half_large_fov[1])*height)
        top = (height - new_height) // 2
        left = (width - new_width) // 2
        output_image = np.zeros_like(input_image, dtype=float)
        output_image[top:top+new_height, left:left+new_width] = transform.resize(input_image,(new_height, new_width))
    else:
        new_width = int(np.tan(half_small_fov[0]) / np.tan(half_large_fov[0]) * width)
        new_height = int(np.tan(half_small_fov[1]) / np.tan(half_large_fov[1]) * height)
        output_image = transform.resize(input_image,(new_height, new_width))
    return output_image


def get_tri_quarter(mtx, direction_key):
    result_mtx = np.zeros_like(mtx)
    tri_mtx = np.tri(mtx.shape[0],mtx.shape[1])
    tri_mtx_rot = np.rot90(tri_mtx)
    mask_mtx = np.logical_and(tri_mtx_rot, tri_mtx) # down
    if direction_key == "down":
        result_mtx[mask_mtx] = mtx[mask_mtx]
    elif direction_key == "right":
        mask_mtx = np.rot90(mask_mtx,1)
        result_mtx[mask_mtx] = mtx[mask_mtx]
    elif direction_key == "up":
        mask_mtx = np.rot90(mask_mtx,2)
        result_mtx[mask_mtx] = mtx[mask_mtx]
    elif direction_key == "left":
        mask_mtx = np.rot90(mask_mtx,3)
        result_mtx[mask_mtx] = mtx[mask_mtx]
    return result_mtx


def get_rec_quarter(mtx, direction_key):
    result_mtx = np.zeros_like(mtx)
    height, width = mtx.shape[:2]
    half_h = int(height/2)
    half_w = int(width/2)
    if direction_key == "down":
        result_mtx[half_h:, :] = mtx[half_h:, :]
    elif direction_key == "right":
        result_mtx[:, half_w:] = mtx[:, half_w:]
    elif direction_key == "up":
        result_mtx[:half_h, :] = mtx[:half_h, :]
    elif direction_key == "left":
        result_mtx[:, :half_w] = mtx[:, :half_w]
    return result_mtx





