import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os
import pandas as pd

from scipy.ndimage.morphology import binary_erosion

from util.unet_triclass_whole_image import unet
import util.od_coords as odc
import util.util as ut

# Turn interactive plotting off
plt.ioff()


# #### TESTING ON SPLIT 1 OF MESSIDOR
data_dir = '../messidor_od_fovea/split1/resized/test/'

masks_dir = '../messidor_od_fovea/split1/test/'

test_dir = os.path.join(data_dir, 'images/')

model_dir = 'results/Messidor_split1/messidor_split1/'

weights_file = 'best_weights.h5'

# # Define model and load weights
model = unet(3, 512, drop=0.)
m1 = model.get_unet(nf=8)  # u net using upsample
m1.load_weights(weights_file)

resized_coordinates = {'Image_File_Name': [],
                       'fovea-x-coordinate': [], 'fovea-y-coordinate': [],
                       'od_x_coords': [], 'od_y_coords': []}

ut.create_dir('results/Preds_test_decay7/')

for fl in sorted(os.listdir(test_dir)):

    resized_coordinates['Image_File_Name'].append(fl)

    # Predict peak location (in 512x512)
    img = io.imread(test_dir + fl)
    img_to_pred = ((img_to_pred - img_to_pred.mean(axis=(0,1))) /
                   (img_to_pred.std(axis=(0,1))))

    dist_map_pred = m1.predict(img_to_pred[np.newaxis,:,:,:])
    pred_map = dist_map_pred[0,:,:,0]

    peak_coords = odc.get_peak_coordinates(pred_map, threshold=0.2)

    od_coords, fov_coords = odc.determine_od(img_to_pred, peak_coords, neigh=9)

    print(od_coords, fov_coords)

    # uncomment to save the predictions to disk
    # io.imsave(model_dir + 'Preds_maps_decay7/' + fl, pred_map, cmap='gray')

    plt.imshow(pred_map)
    plt.plot(od_coords[1], od_coords[0], 'b.')
    plt.plot(fov_coords[1], fov_coords[0], 'r.')

    # Get the locations in the original coordinates (1488x2240)
    fl_img = fl[:-4] + '.tif'
    fl_msk = fl[:-4] + '_test_mask.gif'
    img = io.imread(masks_dir + 'images/' + fl_img)/255.
    mask = io.imread(masks_dir + 'masks/' + fl_msk)/255.
    mask = binary_erosion(mask, structure=np.ones((10, 10)))

    # crop the images to FOV

    b = np.nonzero(mask)
    up, bot = np.min(b[0]), np.max(b[0])
    left, right = np.min(b[1]), np.max(b[1])
    cr_img = img[up:bot, left:right]

    sh_crop = cr_img.shape

    # Resize the predictions to the shape of the crop
    # cr_img_rz = resize(img, img_to_pred)

    od_resh = odc.get_new_peaks(od_coords, sh_crop[:2])
    f_resh = odc.get_new_peaks(fov_coords, sh_crop[:2])

    od_resh = (od_resh[0] + up, od_resh[1] + left)
    f_resh = (f_resh[0] + up, f_resh[1] + left)

    resized_coordinates['fovea-y-coordinate'].append(f_resh[0])
    resized_coordinates['fovea-x-coordinate'].append(f_resh[1])
    resized_coordinates['od_y_coords'].append(od_resh[0])
    resized_coordinates['od_x_coords'].append(od_resh[1])

    fig, ax = plt.subplots(1,2, figsize=(15,10))
    ax[0].imshow(img)
    ax[0].plot(od_resh[1], od_resh[0], 'b.')
    ax[0].plot(f_resh[1], f_resh[0], 'r.')

    ax[1].imshow(pred_map)
    ax[1].plot(od_coords[1], od_coords[0], 'b.')
    ax[1].plot(fov_coords[1], fov_coords[0], 'r.')

    # uncomment to save the predictions to disk
    # plt.savefig(model_dir + 'Preds_test_decay7/' + fl)
    # plt.close('all')

coords_df = pd.DataFrame(data=resized_coordinates)

coords_df.to_csv(model_dir + 'Preds_test_decay7/fovea_od_preds.csv')
