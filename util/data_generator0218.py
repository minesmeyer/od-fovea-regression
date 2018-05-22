"""
Iterator to load images from the datasets, and related functions.
"""
import os
import numpy as np
import pandas as pd

from skimage import io
from skimage import img_as_float
from skimage.transform import resize
from scipy.ndimage import distance_transform_edt

import pickle

from keras import backend as K
from keras.preprocessing.image import Iterator
from keras.preprocessing.image import apply_transform, flip_axis
from keras.preprocessing.image import transform_matrix_offset_center

def normalize_for_tanh(batch):
    """Make input image values lie between -1 and 1."""
    tanh_batch = batch - np.max(batch)/2.
    tanh_batch /= np.max(batch)/2.
    return tanh_batch

class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time, while applying desired
    transformations online."""

    def __init__(self, directory, a_dir_name='A', b_dir_name=None, N=-1,
                 batch_size=32, shuffle=True, seed=None, target_size=(512,512),
                 cspace='rgb', nch_gdt=1,
                 zscore=True, normalize_tanh=False,
                 return_mode='normal', decay=5, dataset='idrid',
                 rotation_range=0., height_shift_range=0., shear_range=0.,
                 width_shift_range=0., zoom_range=0., fill_mode='constant',
                 cval=0., horizontal_flip=False, vertical_flip=False):

        """
        Iterate through the image directoriy, apply transformations and return
        distance map calculated on the fly. If b_dir_name is not None, it will
        retrieve the ground truth from the directory.

        Files under the directory A and B will be returned at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator;
        - return_mode: 'normal', 'fnames'. Default: 'normal'
            - 'normal' returns: [batch_a, batch_b]
            - 'fnames' returns: [batch_a, batch_b, files]
        - decay: decay at which to compute de distance map. Default: 5
        - dataset: dataset to load. Can handle Messidor and Idrid. Default: Idrid

        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.a_fnames = sorted(os.listdir(self.a_dir))

        self.b_dir_name = b_dir_name
        if b_dir_name is not None:
            self.b_dir = os.path.join(directory, b_dir_name)
            self.b_fnames = sorted(os.listdir(self.b_dir))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            self.filenames = self.a_fnames[:N]
        self.N = len(self.a_fnames)

        self.ch_order = K.image_dim_ordering()

        # Preprocess images
        self.cspace = cspace #colorspace

        # Image shape
        self.target_size = target_size
        self.nch_gdt = nch_gdt

        self.nch = len(self.cspace) # for example if grayscale

        self.select_vessels = select_vessels

        self.img_shape_a = self._get_img_shape(self.target_size, ch=self.nch)
        self.img_shape_b = self._get_img_shape(self.target_size, ch=self.nch_gdt)

        if self.ch_order == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2
        else:
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3

        #Normalizations
        self.normalize_tanh = normalize_tanh
        self.zscore = zscore

        # Transformations
        self.rotation_range = rotation_range
        self.height_shift_range = height_shift_range
        self.width_shift_range = width_shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]


        self.return_mode = return_mode

        self.decay=decay
        self.dataset = dataset

        super(TwoImageIterator, self).__init__(len(self.a_fnames), batch_size,
                                               shuffle, seed)

    def _get_img_shape(self, size, ch=3):

        if self.ch_order == 'tf':
            img_shape = size + (ch,)
        else:
            img_shape = (ch,) + size

        return img_shape.

    def _load_img_pair(self, idx):
    """
    Load images and apply pre-processing
    :param idx: index of file to load in the list of names
    :return: aa: image
             bb: ground truth
    """
    aa = img_as_float(io.imread(os.path.join(self.a_dir, self.a_fnames[idx])))
    bb = img_as_float(io.imread(os.path.join(self.b_dir, self.b_fnames[idx])))

    if self.nch_gdt == 3:
        # fix for the case when the .png has an alpha channel
        if bb.shape[-1] == 4:
            bb = bb[:,:,:3]
    elif self.nch_gdt == 1:
        # fix for the case when the .png has an alpha channel
        if len(bb.shape) == 2:
            bb = bb[:,:,np.newaxis]

    if self.select_vessels is True:
        bb = self.select_vessel_width(bb)

    return aa, bb

    def _random_transform(self, a, b, is_batch=False):
        """
        Random dataset augmentation.

        Adapted from https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
        """
        if is_batch is False:
        # a and b are single images, so they don't have image number at index 0
            img_row_index = self.row_index - 1
            img_col_index = self.col_index - 1
            img_channel_index = self.channel_index - 1
        else:
            img_row_index = self.row_index
            img_col_index = self.col_index
            img_channel_index = self.channel_index
            # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
                                                    self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) \
                 * a.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) \
                 * a.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix),
                                  zoom_matrix)

        h, w = a.shape[img_row_index], a.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        a = apply_transform(a, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        b = apply_transform(b, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_col_index)
                b = flip_axis(b, img_col_index)


        if self.vertical_flip:
            if np.random.random() < 0.5:
                a = flip_axis(a, img_row_index)
                b = flip_axis(b, img_row_index)

        return a, b

    def get_dist_maps(self, coords, shp=(512,512)):
        fx, fy = coords[0]
        odx, ody = coords[1]

        distance = np.ones(shp)
        distance[fy, fx] = 0
        distance[ody, odx] = 0
        distance = distance_transform_edt(distance)
        distance = distance[:,:,np.newaxis]
        if shp != (512,512):
            distance=resize(1 - distance / np.max(distance), (512,512,1)) ** self.decay
        else:
            distance = (1 - distance / np.max(distance)) ** self.decay
        return distance

    def next(self):
        """Get the next pair of the sequence."""

        # Lock the iterator when the index is changed.
        with self.lock:
            index_array = next(self.index_generator)
        current_batch_size = len(index_array)

        # Initialize the arrays according to the size of the output images
        batch_a = np.zeros((current_batch_size,) + self.img_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.img_shape_b[:-1]
                           + (self.nch_gdt,))

        files = []
        ind = []

        if self.b_dir_name is None:
            if self.dataset == 'messidor':
                ### For Messidor
                all_coords = pickle.load(open(os.path.join(self.directory + 'resized_coords.pkl'), 'r'))

            elif self.dataset == 'idrid':
                ### For IDRiD
                file_csv_od = os.path.join(self.directory + 'IDRiD_OD_Center_Training_set.csv')
                file_csv_fov = os.path.join(self.directory + 'IDRiD_Fovea_Center_Training_set.csv')

                gt_fovea = pd.read_csv(file_csv_fov)
                # get rid of garbage data
                gt_fovea.drop(gt_fovea.columns[3:], axis=1, inplace=True)
                gt_fovea.drop(gt_fovea.index[413:], inplace=True)

                gt_od = pd.read_csv(file_csv_od)
                # get rid of garbage data
                gt_od.drop(gt_od.columns[3:], axis=1, inplace=True)
                gt_od.drop(gt_od.index[413:], inplace=True)

        # Load images and apply transformations
        for i, j in enumerate(index_array):
            im_id = self.a_fnames[j][:-4]

            if self.b_dir_name is not None:
                a_img, b_img = self._load_img_pair(j)

            else:
                a_img = img_as_float(io.imread(os.path.join(self.a_dir, self.a_fnames[j])))

                if self.dataset == 'messidor':
                    ### For Messidor
                    a_idx = np.where(np.array(all_coords['Image']) == im_id + '.tif')[0][0]
                    coords = [all_coords['fovea'][a_idx], all_coords['od'][a_idx]]
                    # get the distance maps
                    b_img = self.get_dist_maps(coords)

                elif self.dataset == 'idrid':
                    ### For IDRiD
                    fovea_coords = gt_fovea[gt_fovea['Image No'] == im_id]
                    fx, fy = int(fovea_coords['X- Coordinate']), int(fovea_coords['Y - Coordinate'])
                    od_coords = gt_od[gt_od['Image No'] == im_id]
                    odx, ody = int(od_coords['X- Coordinate']), int(od_coords['Y - Coordinate'])
                    coords = [(fx,fy), (odx, ody)]
                    b_img = self.get_dist_maps(coords, shp=(2848, 4288))

            a_img, b_img = self._random_transform(a_img, b_img)
            if self.zscore is True:
                a_img = (a_img - a_img.mean()) / (a_img.std())

            batch_a[i] = a_img
            batch_b[i] = b_img

            files.append(self.a_fnames[j])

        # when using tanh activation the inputs must be between [-1 1]
        if self.normalize_tanh is True and self.zscore is False:
            batch_a = normalize_for_tanh(batch_a)
            batch_b = normalize_for_tanh(batch_b)

        if self.return_mode == 'normal':
            return [batch_a, batch_b]

        elif self.return_mode == 'fnames':
            return [batch_a, batch_b, files]
