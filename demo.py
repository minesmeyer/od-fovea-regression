""" Example run od the OD and Fovea detector """

#!/usr/bin/python
import sys, getopt


import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float, transform
import os
import sys

from util.unet_triclass_whole_image import unet
import util.od_coords as odc


image_dir = '/home/mmeyer/Documents/Projects/Retina/Datasets/DRIDB/AnonimDatabase2/OrigImages/OrigImg0006.bmp'
# mask_dir = '/home/mmeyer/Documents/Projects/Retina/Datasets/MESSIDOR_Fovea/' \
#            'fov_masks/20060407_43436_0200_PP_test_mask.gif'


def demo(image_dir):


    ## Define the model and load the pre-trained weights
    weights_file = 'best_weights.h5'

    model = unet(3, 512, drop=0.)

    m1 = model.get_unet(nf=8)  # u net using upsample
    m1.load_weights(weights_file)

    ## Load the image

    img = img_as_float(io.imread(image_dir))

    img_to_pred = transform.resize(img, (512,512), order=0, mode='constant')
    img_to_pred = (img_to_pred - img_to_pred.mean(axis=(0,1))) / (img_to_pred.std(axis=(0,1)))

    ## Get the location prediction

    dist_map_pred = m1.predict(img_to_pred[np.newaxis,:,:,:])
    pred_map = dist_map_pred[0,:,:,0]

    ## Get the OD and Fovea locations from this distance map

    peak_coords = odc.get_peak_coordinates(pred_map, threshold=0.2)
    od_coords, fov_coords = odc.determine_od(img, peak_coords, neigh=12)

    ## Get the coordinated in the original resolution
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

if __name__ == "__main__":
   demo(sys.argv[1])