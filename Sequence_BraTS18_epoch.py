"""
Pytorch framework for Semi-supervised learning in Medical Image Analysis

MSSL

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""
import Network_training_SSL_epoch_BraTS18
import module.common_module as cm
from dataloader import BraTS18_dataloader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

TSNE = False

data_seed_list = [
                  1,
                  2,
                  3,
                  4,
                  5
                  ]

job_list = [
            'CNN_baseline',
            'MSSL_pretrain_Decoder',
            'MSSL_pretrain_CNN',
            'MSSL_alter_rec',
            'MSSL_joint',
]

data_split_list = [
                   '10L110U',
                   # '20L100U',
                   # '50L70U',
                   # '120L120U',
                   ]


Test_only = False

num_epoch = 252

folder_name = 'results_BraTS/'

Val_dice = 0
val_dice = 0
test_results = 0

for split in data_split_list:
    # Run jobs:
    for job in job_list:
        if job == 'CNN_baseline':
            split = split
            # split = '20L0U'
            Test_results = [0, 0, 0, 0]
            for seed in data_seed_list:
                device, data_sizes, modelDataloader = BraTS18_dataloader.BraTS18data(seed, split)

                if split == '50L70U':
                    num_epoch = 152
                elif split == '120L120U':
                    num_epoch = 52

                val_dice, test_results = Network_training_SSL_epoch_BraTS18.network_training_ssl_epoch(Test_only, job, seed, split, device,
                                                                                               data_sizes, modelDataloader,
                                                                                               num_epoch, folder_name, TSNE)
                Val_dice += val_dice
                for i, item in enumerate(test_results):
                    Test_results[i] += item

                ###################  val each seed
                file = open(str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/val_dice_seed_' + str(
                    seed) + '.txt', 'w')
                title = 'Dice\n'
                history = (
                    '{:4f}\n'
                        .format(Val_dice / len(data_seed_list)))
                file.write(title)
                file.write(history)
                file.close()

                ####################   test each seed
                file = open(
                    str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/test_results_seed_' + str(
                        seed) + '.txt', 'w')
                title = 'Dice               AVD            Recall          F1\n'
                history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(test_results[0], test_results[1], test_results[2], test_results[3]))
                file.write(title)
                file.write(history)
                file.close()

            ####################  overall
            file = open(str(folder_name) + str(job) + '/' + str(split) + '/test_results_overall.txt', 'w')
            title = 'Dice               AVD            Recall          F1\n'
            history = (
                '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                    .format(Test_results[0] / len(data_seed_list), Test_results[1] / len(data_seed_list),
                            Test_results[2] / len(data_seed_list), Test_results[3] / len(data_seed_list)))
            file.write(title)
            file.write(history)
            file.close()

        else:
           # for split in data_split_list[:]:
                Test_results = [0, 0, 0, 0]
                for seed in data_seed_list:
                    device, data_sizes, modelDataloader = BraTS18_dataloader.BraTS18data(seed, split)

                    if split == '50L70U':
                        num_epoch = 152
                    elif split == '120L120U':
                        num_epoch = 52

                    val_dice, test_results = Network_training_SSL_epoch_BraTS18.network_training_ssl_epoch(Test_only, job, seed, split, device,
                                                                                                   data_sizes,
                                                                                                   modelDataloader,
                                                                                                       num_epoch, folder_name, TSNE)
                    Val_dice += val_dice
                    for i, item in enumerate(test_results):
                        Test_results[i] += item

                    ###################  val each seed
                    file = open(
                        str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/val_dice_seed_' + str(
                            seed) + '.txt', 'w')
                    title = 'Dice\n'
                    history = (
                        '{:4f}\n'
                            .format(Val_dice / len(data_seed_list)))
                    file.write(title)
                    file.write(history)
                    file.close()

                    ####################   test each seed
                    file = open(
                        str(folder_name) + str(job) + '/' + str(split) + '/seed' + str(seed) + '/test_results_seed_' + str(
                            seed) + '.txt', 'w')
                    title = 'Dice               AVD            Recall          F1\n'
                    history = (
                        '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                            .format(test_results[0], test_results[1], test_results[2], test_results[3]))
                    file.write(title)
                    file.write(history)
                    file.close()

                ####################  overall
                file = open(str(folder_name) + str(job) + '/' + str(split) + '/test_results_overall.txt', 'w')
                title = 'Dice               AVD            Recall          F1\n'
                history = (
                    '{:4f}        {:.4f}         {:.4f}        {:.4f}\n'
                        .format(Test_results[0] / len(data_seed_list), Test_results[1] / len(data_seed_list),
                                Test_results[2] / len(data_seed_list), Test_results[3] / len(data_seed_list)))
                file.write(title)
                file.write(history)
                file.close()

print('All PC SSL_BraTS jobs finished')
