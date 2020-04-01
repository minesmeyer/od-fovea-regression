""" Example run od the OD and Fovea detector """

#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, transform

from skimage import filters, color
from skimage.measure import regionprops, label
from skimage.morphology import erosion, dilation, disk


from util.unet_triclass_whole_image import unet
import util.od_coords as odc
import util.util as ut

import argparse

parser = argparse.ArgumentParser(description='OD Fovea detection')

parser.add_argument('-i',
                    '--img_dir',
                    help='Image_dir',
                    type=str,
                    default='images/messidor_test.tif')

parser.add_argument('-m',
                    '--mask_dir',
                    help='Mask dir',
                    type=str,
                    default= 'images/messidor_test_mask.tif')

parser.add_argument('-e',
                    '--estimate_fov',
                    action='store_true')

# Define auxiliary functions
def crop_image(img, mask):
    """ Crops the image to the edges of a given FOV mask. """
    b = np.nonzero(mask)
    up, bot = np.amin(b[0]), np.amax(b[0])
    left, right = np.amin(b[1]), np.amax(b[1])
    cr_img = img[up:bot, left:right]
    return cr_img

def get_original_coords(coords, mask):
    """ Get the landmark coordinates for the original resolution. """
    b = np.nonzero(mask)
    up, bot = np.amin(b[0]), np.amax(b[0])
    left, right = np.amin(b[1]), np.amax(b[1])

    new_coords = (coords[0] + up, coords[1] + left)
    return new_coords

def get_mask_fov(image):
    """ Estimate a FOV mask from the original image by simple thresholding. """

    im = color.rgb2gray(image)
    im = filters.rank.enhance_contrast(im, disk(5))
    im = filters.gaussian(im, 2)

    val = filters.threshold_otsu(im)

    mask = np.zeros(im.shape)
    mask[np.where(im < val)] = 1

    # apply closing operation to minimize the contamination with single pixels
    mask = dilation(mask, selem=disk(5))
    mask = erosion(mask, selem=disk(5))

    label_image = label(mask)

    # select the largest connected component as the background
    props = regionprops(label_image)
    pp = [props[i].area for i in range(len(props))]

    mask_new = np.ones(label_image.shape)
    # np.argmax(pp)+1 because label_image also considers 0, but props does not
    mask_new[np.where(label_image == np.unique(label_image)[np.argmax(pp)+1])] = 0

    return mask_new

def demo_od_fovea_detection(args):

    ## Define the model and load the pre-trained weights
    weights_file = 'best_weights.h5'

    model = unet(3, 512, drop=0.)

    m1 = model.get_unet(nf=8)  # u net using upsample
    m1.load_weights(weights_file)

    ## Load the image

    img = img_as_float(io.imread(args.img_dir))

    if args.mask_dir is not None:
        mask = img_as_float(io.imread(args.mask_dir))
        img_crop = crop_image(img, mask)
        img_to_pred = transform.resize(img_crop, (512, 512), order=0, mode='constant')
    elif args.estimate_fov is not False:
        mask = get_mask_fov(img)
        img_crop = crop_image(img, mask)
        img_to_pred = transform.resize(img_crop, (512, 512), order=0, mode='constant')
    else:
        img_to_pred = transform.resize(img, (512,512), order=0, mode='constant')

    img_to_pred = (img_to_pred - img_to_pred.mean(axis=(0,1))) / (img_to_pred.std(axis=(0,1)))

    ## Get the location prediction

    dist_map_pred = m1.predict(img_to_pred[np.newaxis, : :, :])
    pred_map = dist_map_pred[0,:,:,0]

    ## Get the OD and Fovea locations from this distance map

    peak_coords = odc.get_peak_coordinates(pred_map, threshold=0.2)
    od_coords, fov_coords = odc.determine_od(img_to_pred, peak_coords, neigh=12)

    ## Get the coordinates in the original resolution
    if (args.mask_dir is not None) or (args.estimate_fov is not False):
        od_resh = odc.get_new_peaks(od_coords, img_crop.shape[:2])
        f_resh = odc.get_new_peaks(fov_coords, img_crop.shape[:2])

        od_resh = get_original_coords(od_resh, mask)
        f_resh = get_original_coords(f_resh, mask)
    else:
        od_resh = odc.get_new_peaks(od_coords, img.shape[:2])
        f_resh = odc.get_new_peaks(fov_coords, img.shape[:2])

    print('===> OD coordinates: ', od_resh)
    print('===> FOVEA coordinates: ', f_resh)

    fig, ax = plt.subplots(1,2, figsize=(15,10))
    # plt.figure(figsize=(10, 10))
    ax[0].imshow(img)
    ax[0].plot(od_resh[1], od_resh[0], 'b.')
    ax[0].plot(f_resh[1], f_resh[0], 'r.')

    ax[1].imshow(pred_map)
    ax[1].plot(od_coords[1], od_coords[0], 'b.')
    ax[1].plot(fov_coords[1], fov_coords[0], 'r.')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.plot(od_resh[1], od_resh[0], 'b.')
    plt.plot(f_resh[1], f_resh[0], 'r.')
    plt.title('Predicted location of OD (blue) and Fovea (red)')
    plt.xlabel('OD: ( {0}, {1})    Fovea: ({2}, {3}) '.format(od_resh[0], od_resh[1],
                f_resh[0], f_resh[1]))

    ut.create_dir('results/')
    plt.savefig('results/demo.png')


if __name__ == "__main__":
    demo_od_fovea_detection(parser.parse_args())
