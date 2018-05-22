"""Auxiliary methods. """

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def create_dir(mypath):
    """Create a directory if it does not exist."""
    try:
        os.makedirs(mypath)
    except OSError as exc:
        if os.path.isdir(mypath):
            pass
        else:
            raise


def plot_loss(loss, label, filename, log_dir, acc=None, title='', ylim=None):
    """Plot a loss function and save it in a file."""
    loss = np.array(loss)
    plt.figure(figsize=(5, 4))
    plt.plot(loss, label=label)
    if ylim is not None:
        plt.ylim(ylim)
    else:
        if acc is None:
            plt.ylim((0, 0.5))
        else:
            plt.ylim((0,1.))

    plt.title(title)
    plt.savefig(os.path.join(log_dir, filename))
    plt.clf()
    plt.close('all')
