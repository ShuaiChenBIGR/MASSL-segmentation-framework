"""
Pytorch framework for Medical Image Analysis

Data augmentation

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""

from torchvision import transforms, utils
from skimage import io, transform
import numpy as np
import elasticdeform
import torch
import module.common_module as cm
import random
# Transforms


class Resample(object):
    """Resample MRI image from different resolution"""
    def __init__(self, resolution):
        self.resolution = resolution

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        c, z, h, w = image.shape[:]

        new_z, new_h, new_w = int(z * self.resolution[0]), int(h * self.resolution[1]), int(w / self.resolution[2])
        img = transform.resize(image, (c, new_z, new_h, new_w))
        mask = transform.resize(mask, (new_z, new_h, new_w), order=0)
        return {'image': img, 'mask': mask}


class Flip(object):
    """Resample MRI image from different resolution"""
    def __init__(self, frontend=False, horizontal=False, vertical=False):
        self.frontend = frontend
        self.horizontal = horizontal
        self.vertical = vertical

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        seed = random.randint(1, 9999999)
        np.random.seed(seed + 1)

        f_bool = bool(random.getrandbits(1))
        h_bool = bool(random.getrandbits(1))
        v_bool = bool(random.getrandbits(1))

        if self.frontend:
            if f_bool:
                image, mask = image[:, ::-1, :, :], mask[::-1, :, :]
            else:
                image, mask = image[:, :, :, :], mask[:, :, :]

        if self.horizontal:
            if h_bool:
                image, mask = image[:, :, :, ::-1], mask[:, :, ::-1]
            else:
                image, mask = image[:, :, :, :], mask[:, :, :]

        if self.vertical:
            if v_bool:
                image, mask = image[:, :, ::-1, :], mask[:, ::-1, :]
            else:
                image, mask = image[:, :, :, :], mask[:, :, :]

        return {'image': image.copy(), 'mask': mask.copy()}


class Elastic(object):
    def __init__(self, sigma=3, points=(3, 3, 3), order=1):
        self.sigma = sigma
        self.points = points
        self.order = order

    def __call__(self, sample):

        seed = random.randint(1, 9999999)
        np.random.seed(seed + 1)

        elastic_bool = bool(random.getrandbits(1))

        if elastic_bool:
            image, mask = sample['image'], sample['mask']

            [deformed_img, deformed_mask] = elasticdeform.deform_random_grid([image, mask], axis=[(1, 2, 3), (0, 1, 2)],
                                                                             sigma=self.sigma,
                                                                             points=self.points,
                                                                             order=self.order)
            return {'image': deformed_img, 'mask': deformed_mask}

        else:
            return {'image': sample['image'], 'mask': sample['mask']}


class Resize(object):
    """Resize the 3D image [c, z, h, w]
    and 3D mask [z, h, w]
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        c = image.shape[0]
        if isinstance(self.output_size, int):
            new_z, new_h, new_w = self.output_size, self.output_size, self.output_size
        else:
            new_z, new_h, new_w = self.output_size

        new_z, new_h, new_w = int(new_z), int(new_h), int(new_w)

        img = transform.resize(image, (c, new_z, new_h, new_w))
        mask = transform.resize(mask, (new_z, new_h, new_w))

        return {'image': img, 'mask': mask}


class Crop(object):
    """Crop the 3D image [c, z, h, w]
    and 3D mask [z, h, w] in a sample

    Args:
        output_size (tuple or int): desired output size. If int, square crop is made
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        c, z, h, w = image.shape[:]
        new_z, new_h, new_w = self.output_size

        front = int((z - new_z) / 2)
        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        image = image[:, front: front + new_z, top: top + new_h, left: left + new_w]
        mask = mask[front: front + new_z, top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}


class RandomCrop(object):
    """Crop 3D patch randomly from the 3D image [c, z, h, w]
    and 3D mask [z, h, w] in a sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):

        seed = random.randint(1, 9999999)
        np.random.seed(seed + 1)

        image, mask = sample['image'], sample['mask']

        c, z, h, w = image.shape[:]
        new_z, new_h, new_w = self.output_size

        front = np.random.randint(0, z - new_z)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, front: front + new_z, top: top + new_h, left: left + new_w]
        mask = mask[front: front + new_z, top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}


class RandomCropT(object):
    """Crop 3D patch randomly from the 3D image [c, z, h, w]
    and 3D mask [z, h, w] in a sample

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3
            self.output_size = output_size

    def __call__(self, sample):

        seed = random.randint(1, 9999999)
        np.random.seed(seed + 1)

        image, mask = sample['image'], sample['mask']

        c, z, h, w = image.shape[:]
        new_z, new_h, new_w = self.output_size

        front = np.random.randint(0, z - new_z)
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, front: front + new_z, top: top + new_h, left: left + new_w]
        mask = mask[front: front + new_z, top: top + new_h, left: left + new_w]

        return {'image': image, 'mask': mask}


# Convert ndarrays in sample to pytorch tensors

class ToTensor(object):
    """Convert samples to Pytorch tensors
    """

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(mask)}

