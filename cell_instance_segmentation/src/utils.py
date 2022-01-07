from skimage.morphology import remove_small_objects
from skimage.morphology import binary_opening
from skimage.morphology import binary_closing
from skimage.morphology import dilation
from skimage.morphology import erosion

from scipy.ndimage import binary_fill_holes

import scipy.ndimage as ndi

import numpy as np
import yaml

def load_yaml(config_dir):
    """
    Load a yaml config file into a python dictionary.

    """
    with open(config_dir) as cfg_file:
        cfg = yaml.load(cfg_file, Loader = yaml.FullLoader)

    return cfg

def batch_of_n(arr, n):
    """
    Given an array, split into batches of
    size n.

    """
    for i in range(0, len(np_arr), n): 
        yield np_arr[i : i + n]


def n_batches(np_arr, n):
    """
    Given an array, split into n batches.

    """
    return np.array_split(arr, n)


def clean_binary_mask(binary_mask):
    binary_mask = binary_fill_holes(binary_mask)

    binary_mask = remove_small_objects(binary_mask, 150)

    return binary_mask


def remove_overlapping_pixels(mask, other_masks):
    """
    Removes the overlapping pixels in masks as per
    competition rules.

    """
    overlap = np.logical_and(mask, other_masks)

    if np.any(overlap):
        mask[overlap] = 0


def rle_decode(rle_list, shape = (520, 704), arr_type = np.uint16):
    """
    Given a list of RLE encoded masks, decode them and
    return the numpy int64 mask.

    Args:
        rle_list (list): List of RLE encoded masks for
        a whole image.

        shape (tuple): Tuple shape of the final image.
        
    Returns:
        np.ndarray (int 64)

    """
    mask = np.zeros((shape[0] * shape[1], 1), dtype = np.uint64)

    for idx, rle in enumerate(rle_list):
        rle    = rle.split()
        np_rle = np.array(rle, dtype = np.uint64)

        first_indices = np_rle[0 : : 2] - 1 
        lengths       = np_rle[1 : : 2]
        last_indices  = first_indices + lengths 

        for i in range(len(first_indices)):
            mask[first_indices[i] : last_indices[i]] = 1 + idx

    return mask.reshape(shape).astype(arr_type)


def rle_encode(mask_instance):
    """
    Given a single instance of a mask, return the RLE
    encoding of that instance. Will need to be run for
    every instance you may have in your mask.

    Args:
        mask_instance (np.ndarray): A single instance in 
        your mask that needs to be RLE encoded.

    Returns:
        string: The RLE encoding in string form of the mask
        instance

    """
    flattened = mask_instance.flatten()

    shift = np.concatenate([[0], flattened, [0]])

    runs = np.flatnonzero(shift[1:] != shift[:-1]) + 1

    runs[1 : : 2] -= runs[ : : 2]

    return " ".join(str(run) for run in runs)

def mask_bounding_box(mask):
    """
    Returns bounding box given a binary mask. 
    Bounding box given in the form:

        [x_0, y_0, x_1, y_1]

    Where (x_0, y_0) is the top left pixel of the
    box and (x_1, y_1) is the bottom right pixel
    of the box.

    To index a numpy array do:
    np.array[y_0 : y_1, x_0 : x_1]

    Args:
        mask (np.ndarray): Binary Mask

    Returns:
        list: Bounding box of binary mask.

    """
    non_zero_idx = np.where(mask == 1)

    x = non_zero_idx[1]
    y = non_zero_idx[0]

    bounding_box = [
        np.min(x),
        np.min(y),
        np.max(x),
        np.max(y)
    ]

    return bounding_box





