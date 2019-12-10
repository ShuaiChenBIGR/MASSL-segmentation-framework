"""
Pytorch framework for Semi-supervised learning in Medical Image Analysis

BraTS18

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""

import numpy as np
import module.common_module as cm
from glob import glob
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms, utils
import module.transform as trans
import torch


def BraTS18data(data_seed, data_split):

    # Set random seed
    data_seed = data_seed
    np.random.seed(data_seed)

    # Create image list
    HGG_imgList = sorted(glob('data/BraTS2018/HGG/img_*.npy'))
    HGG_maskList = sorted(glob('data/BraTS2018/HGG/mask_*.npy'))

    # Random selection for training, validation and testing
    HGG_list = [list(pair) for pair in zip(HGG_imgList, HGG_maskList)]

    np.random.shuffle(HGG_list)

    train_labeled_img_list = []
    train_labeled_mask_list = []
    train_unlabeled_img_list = []
    train_unlabeled_mask_list = []
    val_labeled_img_list = []
    val_labeled_mask_list = []
    val_unlabeled_img_list = []
    val_unlabeled_mask_list = []
    test_img_list = []
    test_mask_list = []

    if data_split == '20L0U' or data_split == '20L100U':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  20])))
        train_unlabeled_img_list, train_unlabeled_mask_list = map(list, zip(*(HGG_list[20:  120])))
        val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        val_unlabeled_img_list, val_unlabeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        test_img_list, test_mask_list = map(list, zip(*(HGG_list[160:210])))

    elif data_split == '10L0U' or data_split == '10L110U':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  10])))
        train_unlabeled_img_list, train_unlabeled_mask_list = map(list, zip(*(HGG_list[10:  120])))
        val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        val_unlabeled_img_list, val_unlabeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        test_img_list, test_mask_list = map(list, zip(*(HGG_list[160:210])))

    elif data_split == '50L0U' or data_split == '50L70U':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  50])))
        train_unlabeled_img_list, train_unlabeled_mask_list = map(list, zip(*(HGG_list[50:  120])))
        val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        val_unlabeled_img_list, val_unlabeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        test_img_list, test_mask_list = map(list, zip(*(HGG_list[160:210])))

    elif data_split == '120L0U' or data_split == '120L120U':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  120])))
        train_unlabeled_img_list, train_unlabeled_mask_list = map(list, zip(*(HGG_list[0:  120])))
        val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        val_unlabeled_img_list, val_unlabeled_mask_list = map(list, zip(*(HGG_list[120:160])))
        test_img_list, test_mask_list = map(list, zip(*(HGG_list[160:210])))

    elif data_split == 'TSNE':
        train_labeled_img_list, train_labeled_mask_list = map(list, zip(*(HGG_list[0:  1])))
        train_unlabeled_img_list, train_unlabeled_mask_list = map(list, zip(*(HGG_list[0:  1])))
        val_labeled_img_list, val_labeled_mask_list = map(list, zip(*(HGG_list[0:1])))
        val_unlabeled_img_list, val_unlabeled_mask_list = map(list, zip(*(HGG_list[100:101])))
        test_img_list, test_mask_list = map(list, zip(*(HGG_list[0:  50])))

    # Reset random seed
    seed = random.randint(1, 9999999)
    np.random.seed(seed + 1)

    # Build Dataset Class
    class BraTSDataset(Dataset):
        """Segmentation dataset

        Image: T1, Flair [Channel, Z, H, W]
        Labels: 0 - background, 1 - White matter lesion
        """

        def __init__(self, img_list, mask_list, transform=None):
            self.img_list = img_list
            self.mask_list = mask_list
            self.transform = transform

        def __len__(self):
            # assert len(self.img_list) == len(self.mask_list)
            return len(self.img_list)

        def __getitem__(self, idx):
            image = np.load(self.img_list[idx])
            mask = np.load(self.mask_list[idx])

            sample = {'image': image, 'mask': mask}

            if self.transform:
                sample = self.transform(sample)

            return sample


    # Iterating through the dataset
    trainLabeledDataset = BraTSDataset(train_labeled_img_list, train_labeled_mask_list,
                                     transform=transforms.Compose([
                                         trans.RandomCrop(cm.BraTSshape),
                                         # trans.Elastic(),
                                         trans.Flip(horizontal=True),
                                         trans.ToTensor()
                                     ])
                                     )

    trainUnlabeledDataset = BraTSDataset(train_unlabeled_img_list, train_unlabeled_mask_list,
                                       transform=transforms.Compose([
                                           trans.RandomCrop(cm.BraTSshape),
                                           # trans.Elastic(),
                                           trans.Flip(horizontal=True),
                                           trans.ToTensor()
                                       ])
                                       )

    valLabeledDataset = BraTSDataset(val_labeled_img_list, val_labeled_mask_list,
                                   transform=transforms.Compose([
                                       trans.Crop(cm.BraTSshape),
                                       trans.Flip(),
                                       trans.ToTensor()
                                   ])
                                   )

    valUnlabeledDataset = BraTSDataset(val_unlabeled_img_list, val_unlabeled_mask_list,
                                     transform=transforms.Compose([
                                         trans.Crop(cm.BraTSshape),
                                         trans.Flip(),
                                         trans.ToTensor()
                                     ])
                                     )

    testDataset = BraTSDataset(test_img_list, test_mask_list,
                             transform=transforms.Compose([
                                 trans.ToTensor()
                             ])
                             )

    # device_type = 'cpu'
    device_type = 'cuda'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_type)
    dataset_sizes = {'trainLabeled': len(trainLabeledDataset), 'trainUnlabeled': len(trainUnlabeledDataset),
                     'val_labeled': len(valLabeledDataset), 'val_unlabeled': len(valUnlabeledDataset),
                     'test': len(testDataset)}

    modelDataLoader = {'trainLabeled': DataLoader(trainLabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'trainUnlabeled': DataLoader(trainUnlabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'val_labeled': DataLoader(valLabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'val_unlabeled': DataLoader(valUnlabeledDataset, batch_size=1, shuffle=True, num_workers=4),
                       'test': DataLoader(testDataset, batch_size=1, shuffle=True, num_workers=4)}

    return device, dataset_sizes, modelDataLoader
