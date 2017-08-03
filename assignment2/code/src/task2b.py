"""
Task 2 b, visualize the images
"""

import os
import sys
from collections import defaultdict
import argparse
import time
import datetime

import numpy as np
import matplotlib as mpl
#mlp.use('Agg') # To enable matplotlib on server without X
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'viridis'
import tensorflow as tf

import tf_utils2 as tf_utils

from SETTINGS import *

# Constants

U32MODEL = 'task2GRU32st1-lr0.001_bs256_dc0.99_1488480798'
U64MODEL = 'task2GRU64st1-lr0.001_bs256_dc0.99_1488488776'
U128MODEL = 'task2GRU128st1-lr0.001_bs256_dc0.99_1488496514'
U32STACKEDMODEL = 'task2GRU32st3-lr0.001_bs256_dc0.99_1488504944'

U32MODELPATH = os.path.join(PATHS['best_dir'], U32MODEL)
U64MODELPATH = os.path.join(PATHS['best_dir'], U64MODEL)
U128MODELPATH = os.path.join(PATHS['best_dir'], U128MODEL)
U32STACKEDMODELPATH = os.path.join(PATHS['best_dir'], U32STACKEDMODEL)

# Functions

def generate_masked_images(indices=[], seed=0):
    """ sample and mask 100 images from the test sets

    Args:
    indices: (list) list of indices for which images to pick
    seed: (int) numpy seed for reproducing things

    Returns:
    images: (dict) dictionary holding original images and masked images
    indices: (np.array) 1 x 100 indices of the images"""

    # set seed (local to this function so the inpaintings do not depend on
    # this randomness)
    r = np.random.RandomState(seed)

    # Get data set
    data = tf_utils.Data(0)
    test_images, _ = data.get_test_data()
    # If indices are not given we generate them randomly
    if not indices:
        # Create random indices
        indices = r.choice(np.arange(data.test_size), 100, replace=False)
    # Sample images
    sampled_images = test_images[indices[0:100], :]
    # Mask images pas the 300th pixel
    masked_images = sampled_images.copy()
    masked_images[:, -300:] = -1
    masked_images = np.ma.masked_equal(masked_images, -1)

    images = {'masked': masked_images,
              'ground_truth': sampled_images}

    return images, indices


def cross_entropy(y, p):
    """ Calculate the average cross entropy between target and prediction.

    Given the target array y and a prediction array p,
    return the average cross entropy per pixel. For batches it will average
    over batches.

    Args:
    y: (np.array) B x P (batch_size times predicted pixels) array of floats
    representing the target pixels
    p: (np.array) B x P array of the predictions over the pixels

    Returns:
    cross_entropy: (float) average cross entropy over a pixel"""

    # clip to make sure we don't get underflows
    eps = 1e-8
    p = np.clip(p, eps, 1-eps)
    cross_entropy = -(y * np.log(p) + (1 - y) * np.log(1 - p))

    return np.mean(cross_entropy, axis=1)


def fill_in_images(images, name, pixels):
    """ Fill in the images

    Args:
    masked_images: (np.array) B x 784 array of B masked images
    name: (str) name of the model
    pixels: (int) number of pixels to fill in (1, 10, 28, 300)

    Returns:
    sampled_images: (np.array) 10*B x (484 + pixels) predicted images
    image_dict: (dict) dictionary consisting of the original images,
    inpainted images and the Xents"""

    START = 784 - 300 # 784 - 300 - 1

    # Shape of the images
    batch_size, num_pixels = images['masked'].shape

    # temporary image where masks filled in by a value so we can use like normal
    temp_mask = np.asarray(images['masked'])
    temp_mask[temp_mask == -1.0] = 0.5

    # Dictionary holding the different things, 10 for calculating cross entropy
    # only need 5 of these for showing in painting
    sampled_images = {str(i): temp_mask.copy() for i in range(10)}
    sampled_probability = {str(i): np.ones([batch_size, 783]) for i in range(10)}

    # Array for the averaged cross entropies batch_size x 1
    Xent_gt = np.zeros([batch_size, 1])
    Xent_pred = np.zeros([batch_size, 1])

    # We will use the trained model to calculate the pixel probabilities
    with tf.Session() as sess:
        # Need to load the model and necessary operations
        ops = tf_utils.load(sess, name, best_model=True)

        # we need to do it for 10 samples, batch of 100
        for sample in range(10):
            print 'sample {}'.format(sample)
            # we need to fill this in 300 times
            for i in range(0, pixels):
                # For each run we feed in the current
                feed_dict = {ops['input_x']: sampled_images[str(sample)], ops['seq_length']: 784}
                pixel_probs = sess.run(ops['pixel_probs'],
                                       feed_dict=feed_dict)

                # We pick the pixels with regards to the probability at each position
                # then we fill in the mask at position i with black or white depending on
                # the drawn uniform random variable (Here a vector as we do batches)
                u = np.reshape(np.random.uniform(size=[batch_size, 1]), -1)
                sampled_images[str(sample)][:, START + i] = (u < pixel_probs[:, START - 1 + i]).astype(np.float32)

            # Get the probability vector so that we can calculate the cross-entropy
            feed_dict = {ops['input_x']: sampled_images[str(sample)], ops['seq_length']: 784}
            sampled_probability[str(i)] = sess.run(ops['pixel_probs'], feed_dict=feed_dict)
            # get cross entropy for ground truth and predicted image
            # need to fix this
            temp_Xent_gt = np.reshape(cross_entropy(images['ground_truth'][:, START:START+pixels], pixel_probs[:, START-1:START-1+pixels]), [batch_size, 1])
            temp_Xent_pred = np.reshape(cross_entropy(sampled_images[str(sample)][:, START:START+pixels], pixel_probs[:, START-1:START-1+pixels]), [batch_size, 1])
            Xent_gt += temp_Xent_gt
            Xent_pred += temp_Xent_pred

    # Cross-entropies, average it over 10 samples
    Xent_gt /= 10
    Xent_pred /= 10

    print 'All images produced, creating dictionary'

    # Want to create a dictionary such that we have the following structure
    # image_dict is a dictionary. First level image_dict[str(i)] is the key to
    # i'th image. Each such value is a new dictionary containing
    # - filled in image arrays indexed 0 to 9
    # - ground truth image
    # - masked image
    # - ground truth cross entropy
    # - predicted cross entropy

    # defaultdict so that we can create new entries on the fly
    image_dict = defaultdict(dict)

    for i in range(batch_size):
        # key for image i
        key_i = 'image{}'.format(i)
        # ground truth image
        image_dict[key_i]['ground_truth'] = images['ground_truth'][i, :]
        # masked image
        image_dict[key_i]['masked'] = images['masked'][i, :]
        # sampled images
        for sample in range(10):
            image_dict[key_i]['sample{}'.format(sample)] = sampled_images[str(sample)][i, :]
        image_dict[key_i]['Xent_gt'] = Xent_gt[i, 0]
        image_dict[key_i]['Xent_pred'] = Xent_pred[i, 0]

    print 'dictionary produced, returning dictionary'

    return image_dict


def plot_and_save_images(image_dict, name):
    """ Plot and save the images from the image_dict.

    Args:
    image_dict: (dict) dictionary of the necessary information for task2b
    name: (str) name to save the models in

    Returns:

    """

    # Need to loop over the number of images in the batch
    batch_size = len(image_dict)
    # directory where we output
    out_dir = os.path.join(PATHS['save_dir'], name)

    # Save the cross entropies to file
    # need to make sure that directory exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, 'cross_entropies.txt'), 'w') as text_file:
        for i in range(batch_size):
            text_file.write('Index {}. Cross entropy, ground truth: {}, predicted: {}\n'.format(i,
                                                                                              image_dict['image{}'.format(i)]['Xent_gt'],
                                                                                              image_dict['image{}'.format(i)]['Xent_pred']))

        # calculate average Cross entropies over the 100 images
        Xent_gt_avg = 0
        Xent_pred_avg = 0

        for i in range(batch_size):
            Xent_gt_avg += image_dict['image{}'.format(i)]['Xent_gt']
            Xent_pred_avg += image_dict['image{}'.format(i)]['Xent_pred']

        Xent_gt_avg /= batch_size
        Xent_pred_avg /= batch_size

        text_file.write('Average cross entropy, ground truth: {}, predicted: {}\n'.format(Xent_gt_avg, Xent_pred_avg))

    # save images to subfolder
    img_dir = os.path.join(out_dir, 'images')

    # need to make sure that directory exists for images
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Do the same thing but save to individual images
    for i in range(batch_size):
        fig = plt.figure()
        # Plot the ground truth and masked images (column 0 and 1)
        plt.subplot(1, 7, 1)
        plt.imshow(np.reshape(image_dict['image{}'.format(i)]['ground_truth'], newshape=[28, 28]),
                   interpolation='none')
        #plt.title('index {}'.format(i))
        plt.axis('off')

        plt.subplot(1, 7, 2)
        plt.imshow(np.reshape(image_dict['image{}'.format(i)]['masked'], newshape=[28, 28]),
                   interpolation='none')
        plt.axis('off')

        # Plot the predicted sampled images (column 2 to 6)
        for j in range(5):
            plt.subplot(1, 7, 3 + j)
            plt.imshow(np.reshape(image_dict['image{}'.format(i)]['sample{}'.format(j)], newshape=[28, 28]),
            interpolation='none')
            plt.axis('off')

        plt.savefig(os.path.join(img_dir, 'index_{}.png'.format(i)), bbox_inches='tight')
        plt.close('all')

    # Save a 10 x 10 image of all the images, we only take the first image of the samples
    fig = plt.figure()
    gs = mpl.gridspec.GridSpec(10, 10)

    for i in range(int(np.sqrt(batch_size))):
        for j in range(int(np.sqrt(batch_size))):
            plt.subplot(gs[i, j])
            plt.imshow(np.reshape(image_dict['image{}'.format(i*10 + j)]['sample0'], newshape=[28, 28]),
                       interpolation='none')
            plt.axis('off')

    plt.savefig(os.path.join(img_dir, 'image_grid.png'), bbox_inches='tight')
    plt.close('all')

    # save the original and masked grid to be able to see it properly
    fig = plt.figure()

    gs = mpl.gridspec.GridSpec(10, 10)

    for i in range(int(np.sqrt(batch_size))):
        for j in range(int(np.sqrt(batch_size))):
            plt.subplot(gs[i, j])
            plt.imshow(np.reshape(image_dict['image{}'.format(i*10 + j)]['ground_truth'], newshape=[28, 28]),
                       interpolation='none')
            plt.axis('off')

    plt.savefig(os.path.join(img_dir, 'gt_grid.png'), bbox_inches='tight')
    plt.close('all')

    gs = mpl.gridspec.GridSpec(10, 10)

    for i in range(int(np.sqrt(batch_size))):
        for j in range(int(np.sqrt(batch_size))):
            plt.subplot(gs[i, j])
            plt.imshow(np.reshape(image_dict['image{}'.format(i*10 + j)]['masked'], newshape=[28, 28]),
                       interpolation='none')
            plt.axis('off')

    plt.savefig(os.path.join(img_dir, 'masked_grid.png'), bbox_inches='tight')
    plt.close('all')


def main(args):
    # command line arguments
    model = args.model
    if model.lower() == '32':
        name = U32MODEL
    elif model.lower() == '64':
        name = U64MODEL
    elif model.lower() == '128':
        name = U128MODEL
    elif model.lower() == '32st3':
        name = U32STACKEDMODEL
    pixels = args.pixels

    # Sample 100 images from test set
    images, indices = generate_masked_images()

    # Fill in images
    image_dict = fill_in_images(images, name, pixels)

    # Plot and save images
    plot_and_save_images(image_dict, 'task2b{}pixels{}'.format(model, pixels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # OS parameters
    parser.add_argument("-m",
                        "--model",
                        dest="model",
                        help="Specify the model that you want to generate images and Xentropies for",
                        default="32",
                        type=str)
    parser.add_argument("-p",
                        "--pixels",
                        dest="pixels",
                        help="Number of pixels to fill in",
                        default=300,
                        type=int)
    args = parser.parse_args()

    # Run model
    main(args)
