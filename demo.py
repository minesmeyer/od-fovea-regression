""" Example run od the OD and Fovea detector """

#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, transform

from util.unet_triclass_whole_image import unet
import util.od_coords as odc

import argparse

parser = argparse.ArgumentParser(description='OD Fovea detection')

parser.add_argument('-i',
                    '--img_dir',
                    help='Image_dir',
                    type=str,
                    default='messidor_example.tif')

parser.add_argument('-m',
                    '--mask_dir',
                    help='Mask dir',
                    type=str,
                    default=None)

def crop_image(img, mask):
    # Crop the image
    b = np.nonzero(mask)
    up, bot = np.amin(b[0]), np.amax(b[0])
    left, right = np.amin(b[1]), np.amax(b[1])
    cr_img = img[up:bot, left:right]
    return cr_img

def get_original_coords(coords, mask):
    b = np.nonzero(mask)
    up, bot = np.amin(b[0]), np.amax(b[0])
    left, right = np.amin(b[1]), np.amax(b[1])

    new_coords = (coords[0] + up, coords[1] + left)
    return new_coords


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
    else:
        img_to_pred = transform.resize(img, (512,512), order=0, mode='constant')

    img_to_pred = (img_to_pred - img_to_pred.mean(axis=(0,1))) / (img_to_pred.std(axis=(0,1)))

    ## Get the location prediction

    dist_map_pred = m1.predict(img_to_pred[np.newaxis, : :, :])
    pred_map = dist_map_pred[0,:,:,0]

    ## Get the OD and Fovea locations from this distance map

    peak_coords = odc.get_peak_coordinates(pred_map, threshold=0.2)
    od_coords, fov_coords = odc.determine_od(img, peak_coords, neigh=12)

    ## Get the coordinated in the original resolution
    if args.mask_dir is not None:
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
    plt.savefig('results/demo.tif')


if __name__ == "__main__":

    demo_od_fovea_detection(parser.parse_args())
