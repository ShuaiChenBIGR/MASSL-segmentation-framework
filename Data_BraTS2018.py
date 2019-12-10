"""
Pytorch framework for Medical Image Analysis

Create data

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""

from __future__ import print_function, division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from glob import glob
import SimpleITK as sitk

import module.common_module as mkd
import warnings
warnings.filterwarnings('ignore')
matplotlib.use("TkAgg")

plt.ion()

# Data_root_dir = os.path.join(os.path.abspath(__file__), '60_WMH')
Patient_dir = sorted(glob('Original_data/BraTS2018/HGG/*'))

print('-' * 30)
print('Loading files...')
print('-' * 30)

num = 220
num_modality = 4

for nb_file in range(len(Patient_dir)):

    # Set image path
    T1File = sorted(glob(Patient_dir[nb_file] + '/Brats18_*_t1.nii.gz'))
    T2File = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_t2.nii.gz"))
    FLAIRFile = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_flair.nii.gz"))
    T1cFile = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_t1ce.nii.gz"))

    maskFile = sorted(glob(Patient_dir[nb_file] + "/Brats18_*_seg.nii.gz"))


    # Read T1 image
    T1Image = sitk.ReadImage(T1File[0])
    T1Vol = sitk.GetArrayFromImage(T1Image).astype(float)
    T1mask = np.where(T1Vol >= 30, 1, 0)

    T1Vol = T1Vol * T1mask

    # Resample image
    # z, h, w = T1Vol.shape[:]
    # resolution = (3.000, 0.977, 1.200)
    # new_z, new_h, new_w = int(z * resolution[0]), int(h * resolution[1]), int(w / resolution[2])
    # T1Vol = transform.resize(T1Vol, (new_z, new_h, new_w))

    # Set padding and cut parameters
    cut_slice = int(0)

    size_h = 200
    size_w = 200

    if (T1Vol.shape[1] - size_h) % 2 == 0:
        crop_h1 = int((T1Vol.shape[1] - size_h)/2)
        crop_h2 = int((T1Vol.shape[1] - size_h)/2)
    else:
        crop_h1 = int((T1Vol.shape[1] - size_h)/2)
        crop_h2 = int((T1Vol.shape[1] - size_h)/2 + 1)

    if (T1Vol.shape[2] - size_w) % 2 == 0:
        crop_w1 = int((T1Vol.shape[2] - size_w)/2)
        crop_w2 = int((T1Vol.shape[2] - size_w)/2)
    else:
        crop_w1 = int((T1Vol.shape[2] - size_w)/2)
        crop_w2 = int((T1Vol.shape[2] - size_w)/2 + 1)

    # Cropping and cut image
    T1Vol = T1Vol[cut_slice:, crop_h1: (T1Vol.shape[1] - crop_h1), crop_w1:(T1Vol.shape[2] - crop_w2)]

    # Gaussion Normalization
    # T1Vol -= np.mean(T1Vol)
    # T1Vol /= np.std(T1Vol)

    T1Vol *= (1.0 / T1Vol.max())


    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T1 image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     T1slice = T1Vol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(T1slice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    # Read T2 image
    T2Image = sitk.ReadImage(T2File[0])
    T2Vol = sitk.GetArrayFromImage(T2Image).astype(float)
    T2mask = np.where(T2Vol >= 30, 1, 0)

    T2Vol = T2Vol * T2mask

    # Resample image
    # z, h, w = T2Vol.shape[:]
    # resolution = (3.000, 0.977, 1.200)
    # new_z, new_h, new_w = int(z * resolution[0]), int(h * resolution[1]), int(w / resolution[2])
    # T2Vol = transform.resize(T2Vol, (new_z, new_h, new_w))

    # Set padding and cut parameters
    cut_slice = int(0)

    size_h = 200
    size_w = 200

    if (T2Vol.shape[1] - size_h) % 2 == 0:
        crop_h1 = int((T2Vol.shape[1] - size_h)/2)
        crop_h2 = int((T2Vol.shape[1] - size_h)/2)
    else:
        crop_h1 = int((T2Vol.shape[1] - size_h)/2)
        crop_h2 = int((T2Vol.shape[1] - size_h)/2 + 1)

    if (T2Vol.shape[2] - size_w) % 2 == 0:
        crop_w1 = int((T2Vol.shape[2] - size_w)/2)
        crop_w2 = int((T2Vol.shape[2] - size_w)/2)
    else:
        crop_w1 = int((T2Vol.shape[2] - size_w)/2)
        crop_w2 = int((T2Vol.shape[2] - size_w)/2 + 1)

    # Cropping and cut image
    T2Vol = T2Vol[cut_slice:, crop_h1: (T2Vol.shape[1] - crop_h1), crop_w1:(T2Vol.shape[2] - crop_w2)]

    # Gaussion Normalization
    # T2Vol -= np.mean(T2Vol)
    # T2Vol /= np.std(T2Vol)

    T2Vol *= (1.0 / T2Vol.max())


    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T2 image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     T2slice = T2Vol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(T2slice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()


    #  Read FLAIR image
    FLAIRImage = sitk.ReadImage(FLAIRFile[0])
    FLAIRVol = sitk.GetArrayFromImage(FLAIRImage).astype(float)
    FLAIRmask = np.where(FLAIRVol >= 30, 1, 0)

    FLAIRVol = FLAIRVol * FLAIRmask

    # Resample image
    # z, h, w = T1Vol.shape[:]
    # resolution = (3.000, 0.977, 1.200)
    # new_z, new_h, new_w = int(z * resolution[0]), int(h * resolution[1]), int(w / resolution[2])
    # T1Vol = transform.resize(T1Vol, (new_z, new_h, new_w))

    # Set padding and cut parameters
    cut_slice = int(0)

    size_h = 200
    size_w = 200

    if (FLAIRVol.shape[1] - size_h) % 2 == 0:
        crop_h1 = int((FLAIRVol.shape[1] - size_h)/2)
        crop_h2 = int((FLAIRVol.shape[1] - size_h)/2)
    else:
        crop_h1 = int((FLAIRVol.shape[1] - size_h)/2)
        crop_h2 = int((FLAIRVol.shape[1] - size_h)/2 + 1)

    if (FLAIRVol.shape[2] - size_w) % 2 == 0:
        crop_w1 = int((FLAIRVol.shape[2] - size_w)/2)
        crop_w2 = int((FLAIRVol.shape[2] - size_w)/2)
    else:
        crop_w1 = int((FLAIRVol.shape[2] - size_w)/2)
        crop_w2 = int((FLAIRVol.shape[2] - size_w)/2 + 1)

    # Cropping and cut image
    FLAIRVol = FLAIRVol[cut_slice:, crop_h1: (FLAIRVol.shape[1] - crop_h1), crop_w1:(FLAIRVol.shape[2] - crop_w2)]

    # Gaussion Normalization
    # FLAIRVol -= np.mean(FLAIRVol)
    # FLAIRVol /= np.std(FLAIRVol)

    FLAIRVol *= (1.0 / FLAIRVol.max())

    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T1 image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     FLAIRslice = FLAIRVol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(FLAIRslice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()


    #  Read T1c image
    T1cImage = sitk.ReadImage(T1cFile[0])
    T1cVol = sitk.GetArrayFromImage(T1cImage).astype(float)
    T1cmask = np.where(T1cVol >= 30, 1, 0)

    T1cVol = T1cVol * T1cmask

    # Resample image
    # z, h, w = T1Vol.shape[:]
    # resolution = (3.000, 0.977, 1.200)
    # new_z, new_h, new_w = int(z * resolution[0]), int(h * resolution[1]), int(w / resolution[2])
    # T1Vol = transform.resize(T1Vol, (new_z, new_h, new_w))

    # Set padding and cut parameters
    cut_slice = int(0)

    size_h = 200
    size_w = 200

    if (T1cVol.shape[1] - size_h) % 2 == 0:
        crop_h1 = int((T1cVol.shape[1] - size_h)/2)
        crop_h2 = int((T1cVol.shape[1] - size_h)/2)
    else:
        crop_h1 = int((T1cVol.shape[1] - size_h)/2)
        crop_h2 = int((T1cVol.shape[1] - size_h)/2 + 1)

    if (T1cVol.shape[2] - size_w) % 2 == 0:
        crop_w1 = int((T1cVol.shape[2] - size_w)/2)
        crop_w2 = int((T1cVol.shape[2] - size_w)/2)
    else:
        crop_w1 = int((T1cVol.shape[2] - size_w)/2)
        crop_w2 = int((T1cVol.shape[2] - size_w)/2 + 1)

    # Cropping and cut image
    T1cVol = T1cVol[cut_slice:, crop_h1: (T1cVol.shape[1] - crop_h1), crop_w1:(T1cVol.shape[2] - crop_w2)]

    # Gaussion Normalization
    # T1cVol -= np.mean(T1cVol)
    # T1cVol /= np.std(T1cVol)

    T1cVol *= (1.0 / T1cVol.max())

    # Visualize image
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("T1 image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     T1cslice = T1cVol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice), fontsize=14)
    #     ax.axis('off')
    #     plt.imshow(T1cslice, cmap='gray')
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    # Read mask file
    maskImage = sitk.ReadImage(maskFile[0])
    maskVol = sitk.GetArrayFromImage(maskImage).astype(float)
    # Only keep WMH label
    maskVol = np.where(maskVol >= 1, 1, 0)

    # Padding and cut image
    maskVol = maskVol[cut_slice:, crop_h1: (maskVol.shape[1] - crop_h1), crop_w1:(maskVol.shape[2] - crop_w2)]

    # Visualize image
    # transparent1 = 0.8
    # transparent2 = 1.0
    # cmap = pl.cm.viridis
    # my_cmap = cmap(np.arange(cmap.N))
    # my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
    # my_cmap = ListedColormap(my_cmap)
    #
    # plt.figure(figsize=(10, 6))
    # plt.suptitle("Mask on FLAIR image of patient {}".format(Patient_dir[nb_file].split('/')[-1]), fontsize=18)
    # start_slice = 75
    # for i in range(0, 6):
    #     FLAIRslice = FLAIRVol[i + start_slice]
    #     maskslice = maskVol[i + start_slice]
    #     ax = plt.subplot(2, 3, i + 1)
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #     ax.set_title('Slice {}'.format(i + start_slice))
    #     ax.axis('off')
    #     plt.imshow(FLAIRslice, cmap='gray', alpha=transparent1)
    #     plt.imshow(maskslice, cmap=my_cmap, alpha=transparent2)
    #     plt.pause(0.001)
    # plt.show()
    # plt.close()

    imageVol = np.concatenate((np.expand_dims(T1Vol, axis=0), np.expand_dims(T2Vol, axis=0), np.expand_dims(FLAIRVol, axis=0), np.expand_dims(T1cVol, axis=0)), axis=0)

    mkd.mkdir('data')
    mkd.mkdir('data/BraTS2018/HGG')
    np.save('data/BraTS2018/HGG/img_%s.npy' % (str(Patient_dir[nb_file].split('Brats18_')[-1])), imageVol)
    np.save('data/BraTS2018/HGG/mask_%s.npy' % (str(Patient_dir[nb_file].split('Brats18_')[-1])), maskVol)

    print('BraTS2018/HGG Image process {}/{} finished'.format(nb_file, len(Patient_dir)))


print('finished')
