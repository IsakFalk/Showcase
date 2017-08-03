"""
Task 3. Inpaint the most probable pixels
"""

import argparse
import time
import datetime
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib as mpl
#mlp.use('Agg') # To enable matplotlib on server without X
import matplotlib.pyplot as plt

import tf_utils2 as tf_utils

from SETTINGS import *


def load_data(one_pixel=True):
    """ Load the .npy files used for task 3.

    Args:
    one_pixel: (bool) boolean specifying if we are using
    one pixel or patch

    Returns:
    masked_images: (np.array) array for the inpainting data set
    gt_images: (np.array) array of ground truth images"""

    if one_pixel:
        npy_data = 'one_pixel_inpainting.npy'
    else:
        npy_data = '2X2_pixels_inpainting.npy'

    # load data set
    dataset = np.load(os.path.join(PATHS['npy_dir'], npy_data))

    return dataset[0], dataset[1]


def combinations(one_pixel=True):
    """ Get all of the combinations of a patch

    Args:
    one_pixel: (bool) says if we are having one or 2x2 patch

    Returns:
    inpaintings: combinations of inpaintings that exists"""

    if one_pixel:
        inpaintings = [0, 1]
    else:
        inpaintings = [np.asarray((x, y, z, q)) for x in [0, 1] for y in [0, 1] for q in [0, 1] for z in [0, 1]]

    return inpaintings


def masked_indices(masked_imgs, gt_images):
    """ Get the masked points of the masked matrix, for the whole set.

    Args:
    masked_imgs: (np.array) (N x 784) masked images
    gt_images: (np.array) (N x 784) ground truth images

    Returns:
    mask: (np.array) (N x 784) with missing pixels set to 1
    """
    assert (masked_imgs.shape == gt_images.shape), '(masked_imgs) and (gt_imgs) must be same shape.'

    return ((masked_imgs - gt_images) != 0).astype(np.float32)


def inpaint(masked_imgs, inpainting):
    """ Insert inpainting values (2x2 or 1 pixel) where we have the masked.
    Since we add this on top of the masked image, we set a 0 to be 1 and a 1 to be 2,
    since the masks are -1, (-1 + 1 = 0, -1 + 2 = 1).

    Args:
    masked_imgs: (np.array) (N x 784) array of the masked images
    inpainting: (np.array) the inpainting that we want to use (1 or 16 of them)

    Returns:
    paint_mask: (np.array) (N x 784) array that when added to masked gives the wanted
    combination
    """
    # dimensions
    N, D = masked_imgs.shape

    # array for holding the inpainted image
    paint_mask = np.zeros((N, D))

    for i in range(N):
        paint_mask[i, masked_imgs[i, :] == 1] = inpainting + 1

    return paint_mask


def task3b(one_pixel=True):
    # Use GRU 128 hidden unit model
    name = U128MODEL

    # Load the data
    masked_imgs, gt_imgs = load_data(one_pixel=one_pixel)

    # Dimensions
    N, D = gt_imgs.shape

    # Get the place where it is masked
    masked_mat = masked_indices(masked_imgs, gt_imgs)

    # get the different inpaintings
    inpaintings = combinations(one_pixel=one_pixel)

    C = len(inpaintings)

    # Hold optimal data
    inpaint_Xent = np.zeros((N, C))

    with tf.Session() as sess:
        ops = tf_utils.load(sess, name, best_model=True)

        # Set the gt cross entropy
        feed_dict = {ops['input_x']: gt_imgs, ops['seq_length']: 784}
        gt_Xent = sess.run(ops['losses'],
                           feed_dict=feed_dict)

        # loop over the inpaintings and get the Xent to get the best ones
        for i, inpaint_patch in enumerate(tqdm(inpaintings)):
            paint_mask = inpaint(masked_mat, inpaint_patch)
            inpainted_imgs = paint_mask + masked_imgs
            feed_dict={ops['input_x']: inpainted_imgs, ops['seq_length']: 784}
            inpaint_Xent[:, i] = sess.run(ops['losses'],
                                          feed_dict=feed_dict)

        # Get the max and argmax for each of these
        opt_inpaint_index = np.argmin(inpaint_Xent, axis=1)
        inpaint_Xent = np.amin(inpaint_Xent, axis=1)

    # Create an array that takes the optimal patches for each image
    opt_inpaint = [inpaintings[i] for i in opt_inpaint_index]

    def fill_out_imgs(masked_imgs, opt_inpaint):
        """ Helper function to fill out all of the images
        with the optimal patches"""

        N, D = masked_imgs.shape
        optimal_inpaintings = masked_imgs.copy()
        for i in range(N):
            optimal_inpaintings[i, (optimal_inpaintings[i, :] == -1)] = opt_inpaint[i]

        return optimal_inpaintings

    # The optimal inpaintings from using looking at the minimizer of the cross entropy
    # among the patches
    optimal_inpaintings = fill_out_imgs(masked_imgs, opt_inpaint)

    # Calculate the difference in the cross entropy
    diff_Xent = inpaint_Xent - gt_Xent

    # Create an image of all the histograms
    fig, ax = plt.subplots(3, 1)

    # Difference
    ax[0].hist(diff_Xent, 100)
    ax[0].set_title('Differences of Cross-Entropies')
    ax[0].set_xlabel('inpainted Xent - ground truth Xent')
    ax[0].set_ylabel('counts')
    ax[0].grid(True)
    ax[0].set_xlim([-0.05, 0.0])

    # Ground Truth
    ax[1].hist(gt_Xent, bins=20)
    ax[1].set_title('Ground Truth Cross-Entropy')
    ax[1].set_xlabel('Ground Truth Xent')
    ax[1].set_ylabel('counts')
    ax[1].grid(True)

    # Inpainted
    ax[2].hist(inpaint_Xent, bins=20)
    ax[2].set_title('Inpainted Cross-Entropy')
    ax[2].set_xlabel('Inpainted Xent')
    ax[2].set_ylabel('counts')
    ax[2].grid(True)

    # tight layout
    fig.tight_layout()

    if one_pixel:
        pix = '1'
    else:
        pix = '2x2'

    # Save figure
    plt.savefig('inpainting_{}_hist.png'.format(pix))
    plt.close('all')

    # Example figures (plot the optimal inpainting, gt, missing images)
    # for the 10 first samples
    fig, axes = plt.subplots(3, 3)

    # do this for the 10 first samples
    for i in range(3):
        axes[i, 0].imshow(gt_imgs[i, :].reshape([28, 28]), interpolation='none')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(masked_imgs[i, :].reshape([28, 28]), interpolation='none')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(optimal_inpaintings[i, :].reshape([28, 28]), interpolation='none')
        axes[i, 2].axis('off')

    # suptitle
    fig.suptitle('Comparison of images')
    # Save figure
    plt.savefig('image_comp_{}pixel.png'.format(pix))
    plt.close('all')

    # print statistics to stdout
    print "Ground truth Cross-entropy"
    print "mean: {}, std: {}".format(gt_Xent.mean(), gt_Xent.std())

    print "Inpainting Cross-entropy"
    print "mean: {}, std: {}".format(inpaint_Xent.mean(), inpaint_Xent.std())

    print "Difference Cross-entropy"
    print "mean: {}, std: {}".format(diff_Xent.mean(), diff_Xent.std())

    # save array
    np.save('optimal_inpaintings_{}pixel_patch'.format(pix), optimal_inpaintings)

    print "Optimal inpaintings saved to disk."


if __name__ == '__main__':
    # Run the script if you want to plot the images,
    # save the .npy arrays and calculate statistics
    task3b(one_pixel=False)
    task3b(one_pixel=True)
