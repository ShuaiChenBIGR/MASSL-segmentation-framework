# MASSL-segmentation-framework

Multi-task Attention-based Semi-supervised Learning framework for image segmentation based on the paper published at MICCAI 2019 (https://arxiv.org/abs/1907.12303) by Shuai Chen, et al.

For questions please contact me through github or email directly.

## Requirements
1. python 3
2. matplotlib
3. numpy
4. SimpleITK
5. sklearn
6. pytorch, torchvision
7. tqdm
8. skimage
9. scipy
10. elasticdeform

## Training steps
1. Prepare 3D data for training, validation, and testing. Set the image patch size in module/common_module.py [BraTSshape]. Set folder path, preprocessing, and save as .npy files in Data_BraTS2018.py.
2. Set dataloader for pytorch, data split, and data augmentation in dataloader/BraTS18_dataloader.py.
3. Set random data seed, job you want to run, and data split you want to test in Sequance_BraTS18_epoch.py [for CNN baseline, pretraining methods, and MSSL method], or Sequance_BraTS18.epoch.py [for MASSL method]. 
4. Run Sequance_BraTS18_epoch.py or Sequance_BraTS18.epoch.py for training.

## Testing
Change variable [Test_only=True] in Sequance_BraTS18_epoch.py or Sequance_BraTS18.epoch.py and run again. 

## Citation
If you find the method useful for your research, please consider citing the paper:

@inproceedings{chen2019multi,
  title={Multi-Task Attention-Based Semi-Supervised Learning for Medical Image Segmentation},
  author={Chen, Shuai and Bortsova, Gerda and Ju{\'a}rez, Antonio Garc{\'\i}a-Uceda and van Tulder, Gijs and de Bruijne, Marleen},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={457--465},
  year={2019},
  organization={Springer}
}

