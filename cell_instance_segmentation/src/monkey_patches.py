# This contains "monkey patches" for the Cellpose Library and Detectron2
# to do the following:
# 
# Cellpose: They hard coded inter = False into their SizeModel and so I 
#           "monkey patched" it and received a very small increase in CV
#           and LB.
# 
# Detectron2: Replaced "to_bitmasks" and "paste_masks_in_image" to get per
#             pixel predictions which also lead to a small increase in CV 
#             and LB.
# 

from detectron2.layers.mask_ops import _do_paste_mask
from cellpose import models, io, plot, utils
from types import SimpleNamespace
from tqdm import tqdm, trange
import numpy as np
import torch


# The following 2 functions allow for per pixel predictions in Detectron2
def to_bitmasks(self, boxes: torch.Tensor, height, width, threshold = 0.5):
        bitmasks = paste_masks_in_image(
            self.tensor,
            boxes,
            (height, width),
            threshold = threshold,
        )
        return SimpleNamespace(tensor = bitmasks)


@torch.jit.script_if_tracing
def paste_masks_in_image(masks, boxes, image_shape, threshold=0.5):
    """
    Copy pasted from detectron2.layers.mask_ops.paste_masks_in_image and deleted thresholding of the mask.
    Removed GPU ability for memory issues. Will slow down code unfortunately.

    """
    assert masks.shape[-1] == masks.shape[-2], "Only square mask predictions are supported"

    N = len(masks)

    if N == 0:
        return masks.new_empty((0,) + image_shape, dtype = torch.uint8)

    if not isinstance(boxes, torch.Tensor):
        boxes = boxes.tensor

    assert len(boxes) == N, boxes.shape

    img_h, img_w = image_shape
    num_chunks   = N
    device       = boxes.device

    chunks = torch.chunk(torch.arange(N, device = device), num_chunks)

    img_masks = torch.zeros(
        N, img_h, img_w, device = device, dtype = torch.float32
    )

    for inds in chunks:
        masks_chunk, spatial_inds = _do_paste_mask(
            masks[inds, None, :, :], boxes[inds], img_h, img_w, skip_empty = device.type == "cpu"
        )

        img_masks[(inds,) + spatial_inds] = masks_chunk

    return img_masks

# I switched this out because they have interp = False hard-coded.
# and interp = True gave a small increase in CV and LB.
def eval(self, x, channels=None, channel_axis=None, 
         normalize=True, invert=False, augment=False, tile=True,
         batch_size=8, progress=None, interp=True, omni=False):

    assert x.squeeze().ndim < 3

    styles = self.cp.eval(x, channels=channels, channel_axis=channel_axis, 
                          normalize=normalize, invert=invert, augment=augment, 
                          tile=tile, batch_size=batch_size, net_avg=True,
                          compute_masks=False)[-1]

    diam_style = self._size_estimation(np.array(styles))
    diam_style = self.diam_mean if (diam_style==0 or np.isnan(diam_style)) else diam_style

    masks = self.cp.eval(x, channels=channels, channel_axis=channel_axis, 
                         normalize=normalize, invert=invert, augment=augment, 
                         tile=tile, batch_size=batch_size, net_avg=True,
                         rescale=(self.diam_mean / diam_style), diameter=None,
                         interp=True, omni=omni)[0]

    # allow backwards compatibility to older scale metric
    diam = utils.diameters(masks, omni = omni)[0]
    if hasattr(self, 'model_type') and (self.model_type=='nuclei' or self.model_type=='cyto') and not self.torch and not omni:
        diam_style /= (np.pi**0.5)/2
        diam = self.diam_mean / ((np.pi**0.5)/2) if (diam==0 or np.isnan(diam)) else diam
    else:
        diam = self.diam_mean if (diam==0 or np.isnan(diam)) else diam

    return diam, diam_style





