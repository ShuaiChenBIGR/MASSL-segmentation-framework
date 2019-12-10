
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator


# Visualize predictions:
def visualize_Seg(img, mask, preds, slice=20, figsize=(10, 6), row=2, col=2, epoch=None):

    fig = plt.figure(num='prediction', figsize=figsize)
    if epoch is not None:
        plt.suptitle('epoch: {}'.format(epoch), fontsize=16)
    cmap = pl.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    ax = plt.subplot(row, col, 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Image', fontsize=14)
    ax.axis('off')
    plt.imshow(img[slice], cmap='gray', alpha=1.0)
    # plt.imshow(mask[slice], cmap=my_cmap, alpha=1.0)

    ax = plt.subplot(row, col, 1 + col)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Prediction', fontsize=14)
    ax.axis('off')
    # plt.imshow(img[slice], cmap='gray', alpha=0.0)
    plt.imshow(preds[slice], cmap='viridis', alpha=1.0)

    ax = plt.subplot(row, col, 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Ground Truth', fontsize=14)
    ax.axis('off')
    plt.imshow(img[slice], cmap='gray', alpha=1.0)
    plt.imshow(mask[slice], cmap=my_cmap, alpha=1.0)

    return fig



# Visualize predictions:
def visualize_Rec(img, mask_1, mask_2, preds_1, preds_2, slice=20, figsize=(10, 6), row=2, col=2, epoch=None):

    fig = plt.figure(num='prediction', figsize=figsize)
    if epoch is not None:
        plt.suptitle('epoch: {}'.format(epoch), fontsize=16)
    cmap = pl.cm.viridis
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    my_cmap = ListedColormap(my_cmap)

    ax = plt.subplot(row, col, 1)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Ground Truth', fontsize=14)
    ax.axis('off')
    # plt.imshow(img[slice], cmap='gray', alpha=1.0)
    plt.imshow(mask_1[slice], cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, 1 + col)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Prediction', fontsize=14)
    ax.axis('off')
    # plt.imshow(img[slice], cmap='gray', alpha=0.0)
    plt.imshow(preds_1[slice], cmap='viridis', alpha=1.0)


    ax = plt.subplot(row, col, 2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Ground Truth', fontsize=14)
    ax.axis('off')
    # plt.imshow(img[slice], cmap='gray', alpha=1.0)
    plt.imshow(mask_2[slice], cmap='gray', alpha=1.0)

    ax = plt.subplot(row, col, 2 + col)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax.set_title('Prediction', fontsize=14)
    ax.axis('off')
    # plt.imshow(img[slice], cmap='gray', alpha=0.0)
    plt.imshow(preds_2[slice], cmap='viridis', alpha=1.0)

    return fig


# Visualize training loss:
def visualize_loss(fig, row=1, col=3, dict=None, title=None, epoch=None):

    train_labeled_scale = 1
    val_labeled_scale = 1
    unlabeled_scale = 1

    if epoch is not None:
        fig.suptitle('epoch: {}'.format(epoch), fontsize=16)

    ax1 = plt.subplot(row, col, 1)
    plt.ylim(0, train_labeled_scale)
    ax1.set_title('training loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('seg loss')
    lns1 = plt.plot(dict[title[0]], dict[title[1]], 'royalblue', label='Seg loss')

    plt.grid(which='major', axis='y', linestyle='--')

    ax2 = ax1.twinx()
    plt.ylim(0, unlabeled_scale)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('rec loss')
    lns2 = plt.plot(dict[title[0]], dict[title[2]], 'crimson', label='Rec loss')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)


    ax1 = plt.subplot(row, col, 2)
    plt.ylim(0, val_labeled_scale)
    ax1.set_title('val loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('seg loss')
    lns1 = plt.plot(dict[title[0]], dict[title[3]], 'royalblue', label='Seg loss')

    plt.grid(which='major', axis='y', linestyle='--')

    ax2 = ax1.twinx()
    plt.ylim(0, unlabeled_scale)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('rec loss')
    lns2 = plt.plot(dict[title[0]], dict[title[4]], 'crimson', label='Rec loss')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    if epoch == 0:
        plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    ax1 = plt.subplot(row, col, 3)
    plt.ylim(0, val_labeled_scale)
    ax1.set_title('val performance')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Dice')
    lns1 = plt.plot(dict[title[0]], dict[title[5]], 'royalblue', label='Seg Dice')

    plt.grid(which='major', axis='y', linestyle='--')

    ax2 = ax1.twinx()
    plt.ylim(0, unlabeled_scale)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('MSE')
    lns2 = plt.plot(dict[title[0]], dict[title[6]], 'crimson', label='Rec MSE')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]

    plt.grid(which='major', axis='y', linestyle='--')
    plt.legend(lns, labs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    return None
