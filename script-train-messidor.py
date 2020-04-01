""" Main script meant for training on Messidor or IDRiD datasets """

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import os

from util.data_generator0218 import TwoImageIterator
import util.util as ut
from util.unet_triclass_whole_image import unet

from keras.optimizers import Adam
from keras import backend as K


def check_EarlyStop(vloss, tr_loss, patience=5):

    Eopt = np.min(vloss[:-1])
    GL = ((vloss[-1] / Eopt) - 1)
    Pk = (np.sum(tr_loss[-patience:]) / (
            patience * (np.min(tr_loss[-patience:]))))
    PQ = GL / Pk

    # return [PQ, GL, Pk]
    if (GL > 0.):
        if PQ > 0.5:
            return 'early_stop'
        elif Pk < 1.1:
            return 'early_stop'
        else:
            return 'pass'
    else:
        return 'pass'


# #### TRAINING ON SPLIT 1 OF MESSIDOR
data_dir = '../messidor_od_fovea/split1/resized/'

a_tr, b_tr = 'train/images/', 'train/gdt/'
a_val, b_val = 'val/images/', 'val/gdt/'

# ## For Messidor
nsamples = 455
nsamples_val = 113

# ## For IDRiD
# nsamples = 373.
# nsamples_val = 40.

nb_epochs = 601
batch_size = 16

b_iter = int(np.ceil(nsamples / batch_size))

val_batch_size = int(np.ceil(nsamples_val/b_iter))

print(b_iter)

decay = 7

# #### Define the Model
model = unet(3, 512, drop=0.5)
m1 = model.get_unet(nf=8)  # u net using upsample
lr = 0.005
adam = Adam(lr=lr)
m1.compile(optimizer=adam, loss='mse')
path_model = '../results/Messidor_split1_NEW/2804_messidor_split1_nf8_decay{0}/'.format(decay)

# #### Set the iterators
train_it = TwoImageIterator(data_dir, a_dir_name=a_tr,
                            b_dir_name=b_tr, N=-1,
                            batch_size=batch_size, shuffle=True, seed=None,
                            target_size=(512,512), nch_gdt=1,
                            rotation_range=0.2, height_shift_range = 0., shear_range = 0.,
                            width_shift_range = 0., zoom_range = 0., fill_mode='constant',
                            cval = 0., horizontal_flip=True, vertical_flip=True,
                            cspace='rgb',
                            normalize_tanh=False, zscore=True, decay=decay, dataset='messidor')

val_it = TwoImageIterator(data_dir, a_dir_name=a_val,
                          b_dir_name=b_val, N=-1,
                          batch_size=val_batch_size, shuffle=True, seed=None,
                          target_size=(512,512), nch_gdt=1,
                          cspace='rgb',
                          normalize_tanh=False, zscore=True, decay=decay, dataset='messidor')

# Create the folder where intermediate results will be saved for verification
tr_folder = 'train_continued/'
ut.create_dir(path_model + tr_folder)

# #### Training Loop
l_ep = {}
acc_ep = {}

losses = {'train': [], 'val': []}
epoch = 0
for e in range(epoch, nb_epochs):

    print('Epoch %d' % (e + 1))
    l_ep['train'] = np.zeros(b_iter)
    l_ep['val'] = np.zeros(b_iter)

    acc_ep['train'] = np.zeros(b_iter)
    acc_ep['val'] = np.zeros(b_iter)

    for it in range(b_iter):
        x_batch, y_batch = next(train_it)
        xval, yval = next(val_it)

        tmp = m1.fit(x_batch, y_batch, batch_size=batch_size, epochs=1,
                     shuffle=True, validation_data=(xval, yval),
                     verbose=0)

        # save loss for further inspection
        l_ep['train'][it] = tmp.history['loss'][0]
        l_ep['val'][it] = tmp.history['val_loss'][0]

        print('batch' + str(it) + ': ')
        print(tmp.history)
        if np.isnan(tmp.history['loss']):
            raise Exception('Loss is NaN')

    val_loss, tr_loss = np.median(l_ep['val']), np.median(l_ep['train'])

    print('loss: [%.6f], val_loss: [%0.6f]' % (tr_loss, val_loss))

    losses['train'].append(np.mean(l_ep['train']))
    losses['val'].append(np.mean(l_ep['val']))

    # Save best model
    if e > epoch+1:

        Eopt = np.min(losses['val'][:-1])

        if losses['val'][-1] < Eopt:
            m1.save((path_model +'best_model.h5'),
                            overwrite=True)
            m1.save_weights((path_model + 'best_weights.h5'),
                                    overwrite=True)

    # save intermediate to folder results every 10 epochs
    if e % 10 == 0:

        ypred = m1.predict(xval)
        x_plt = (xval[0] - xval[0].min()) / (xval[0].max() - xval[0].min())
        fix, ax = plt.subplots(1,3, figsize=(10,10))

        ax[0].imshow(x_plt)
        ax[1].imshow(yval[0, :, :, 0], cmap='gray')
        ax[2].imshow(ypred[0, :, :, 0], cmap='gray')

        plt.savefig((path_model + tr_folder + 'val_pred_e_' + str(e) + '.png'))

        sv_path = os.path.join(path_model, tr_folder)
        pickle.dump(losses, open(sv_path + 'losses.pkl', 'wb'))

        # plot the training and validation losses
        ut.plot_loss(losses['train'], 'Train Loss', 'unet_loss.png',
                     sv_path, title='Training Loss', ylim=(0, 0.05))
        ut.plot_loss(losses['val'], 'Validation Loss', 'unet_loss_val.png',
                     sv_path, title='Validation_Loss', ylim=(0, 0.05))

    if (e >= epoch+50) and (e % 50 == 0):
        if check_EarlyStop(losses['val'], losses['train'], patience=50) == 'early_stop':
            print('Decreasing learning rate to lr= %f' % (lr/2))
            K.set_value(m1.optimizer.lr, lr/2)
        else:
            pass

    if (e >= 100) and (e % 100 == 0):
        if check_EarlyStop(losses['val'], losses['train'], patience=100) == 'early_stop':
            print('Early Stopping Model...')
            break
        else:
            pass
