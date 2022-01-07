import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(image_grey, compare_0, compare_1, actual, show = True):
    """
    Good for visualizing comparisons between mutliple predictions.

    | ======= | ======= | ======= | 
    |  I + 0  |  I + 1  |  I + A  | 
    | ======= | ======= | ======= | 

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.set_title("Prediction 0")
    ax1.imshow(image_grey, cmap = "gray", interpolation = "none")
    ax1.imshow(compare_0,  cmap = "jet",  alpha = 0.3)

    ax2.set_title("Prediction 1")
    ax2.imshow(image_grey, cmap = "gray", interpolation = "none")
    ax2.imshow(compare_1,  cmap = "jet",  alpha = 0.3)

    ax3.set_title("Ground Truth")
    ax3.imshow(image_grey, cmap = "gray", interpolation = "none")
    ax3.imshow(actual,     cmap = "jet",  alpha = 0.3)

    if show: plt.show()


def plot_pred_actual(image, pred, actual, show = True):
    """
    Given image, predicted mask, and the actual mask. Plot
    in this order:

    | ======= | ========== | 
    |    I    |   I + P    | 
    | ======= | ========== | 
    |  I + A  |  I + P + A | 
    | ======= | ========== | 

    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.set_title("Image")
    ax1.imshow(image)

    ax2.set_title("Image & Pred")
    ax2.imshow(image)
    ax2.imshow(pred, alpha = 1)

    ax3.set_title("Image & Actual")
    ax3.imshow(image)
    ax3.imshow(actual, alpha = 0.3)

    ax4.set_title("Image & Pred & Actual")
    ax4.imshow(image)
    ax4.imshow(pred,   alpha = 0.3)
    ax4.imshow(actual, alpha = 0.3)

    if show: plt.show()


def plot_image_mask(image, mask, show = True):
    """
    Given image, predicted mask, and the actual mask. Plot
    in this order:

    | ======= | 
    |    I    |
    | ======= |
    |  I + M  |
    | ======= |

    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.set_title("Image")
    ax1.imshow(image)

    ax2.set_title("Image & Mask")
    ax2.imshow(image)
    ax2.imshow(mask, alpha = 0.5)

    if show: plt.show()


def simple_plot(mask, actual):
    plt.imshow(mask)
    plt.show()

    plt.imshow(actual)
    plt.show()


