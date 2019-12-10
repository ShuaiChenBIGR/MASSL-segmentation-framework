"""
Pytorch framework for Semi-supervised learning in Medical Image Analysis

Training and validation

Author(s): Shuai Chen
PhD student in Erasmus MC, Rotterdam, the Netherlands
Biomedical Imaging Group Rotterdam

If you have any questions or suggestions about the code, feel free to contact me:
Email: chenscool@gmail.com

Date: 22 Jan 2019
"""

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import module.common_module as cm
from module.dice_loss import DiceCoefficientLF, MSELF, DiceCoefficientLF_rec
from module.visualize import visualize, visualize_loss
from module.eval_BraTS_slidingwindow import eval_net_dice, eval_net_mse, test_net_dice

from collections import defaultdict
from network import ssl_3d_sep
import time
import copy
from tqdm import trange

import warnings
warnings.filterwarnings('ignore')


def train_model(model, modelDataLoader, device, root_path, network_switch, criterion, optimizer, scheduler,
                num_epochs=25, jointly=False, self=False, num_optimizer='two', mode='rec', unet=False):
    since = time.time()
    # initialize training parameters
    inputs = 0
    labels = 0
    inputs2 = 0
    labels2 = 0
    outputs = 0

    PREVIEW = True

    dict = defaultdict(list)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_dice = 0.0
    best_val_mse = 1.0
    best_epoch = 0

    epoch_val_loss = np.array([0.0, 1.0])

    epoch_val_dice = 0.0
    epoch_val_mse = 1.0

    w1 = 0
    w2 = 0
    loss = 0

    # set TQDM iterator
    tqiter = trange(num_epochs, desc='BraTS')

    for epoch in tqiter:
    # for epoch in range(num_epochs):

        epoch_train_loss = np.array([0.0, 0.0])
        fig_loss = plt.figure(num='loss', figsize=[12, 3.8])

    # go through all batches
        for i, (sample1, sample2) in enumerate(zip(modelDataLoader['trainLabeled'], modelDataLoader['trainUnlabeled'])):

            # Only loop the smaller dataset (Labeled or Unlabeled) then validate:
            if i < (len(modelDataLoader['trainLabeled']) - 1) and i < (len(modelDataLoader['trainUnlabeled']) - 1):
                procedure = ['trainLabeled', 'trainUnlabeled']
            else:
                procedure = ['trainLabeled', 'trainUnlabeled', 'val_labeled', 'val_unlabeled']

            # run training and validation alternatively:
            for phase in procedure:

                if phase == 'trainLabeled':
                    scheduler[0].step()
                    model.train()
                elif phase == 'trainUnlabeled':
                    scheduler[1].step()
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                # If 'labeled', then use segmentation mask; else use image for reconstruction
                if phase == 'trainLabeled':
                    inputs = sample1['image'][:, 2:3].float().to(device)
                    labels = sample1['mask'].long().to(device)

                    if not self:
                        labels2 = sample2['image'][:, 2].long().to(device)

                elif phase == 'trainUnlabeled':
                    inputs = sample2['image'][:, 2:3].float().to(device)
                    if mode == 'rec':
                        labels = sample2['image'][:, 2].long().to(device)
                    elif mode == 'seg':
                        labels = sample2['mask'].long().to(device)

                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                # update model parameters and compute loss
                with torch.set_grad_enabled(phase == 'trainLabeled' or phase == 'trainUnlabeled'):

                    if phase == 'trainLabeled':
                        outputs = model(inputs, phase=phase, network_switch=network_switch)[0]

                        w1 = 1.0
                        w2 = 0.0

                        if self:
                            loss = w1 * criterion[0](outputs.float(), labels.float()) \
                                    + w2 * criterion[1](outputs.float(), labels.float())
                        else:
                            loss = w1 * criterion[0](outputs.float(), labels.float()) \
                                    + w2 * criterion[1](outputs.float(), labels2.float())

                    elif phase == 'trainUnlabeled':
                        outputs = model(inputs, phase=phase, network_switch=network_switch)[1]
                        # print(outputs.shape)
                        w1 = 0.0
                        w2 = 1.0

                        loss = w1 * criterion[0](outputs.float(), labels.float()) \
                               + w2 * criterion[1](outputs.float(), labels.float())

                    outputs_vis = outputs.cpu().detach().numpy()
                    inputs_vis = inputs.cpu().detach().numpy()
                    labels_vis = labels.cpu().detach().numpy()

                    # visualize training set at the end of each epoch
                    if PREVIEW:
                        if i == (len(modelDataLoader['trainLabeled']) - 1):
                            if phase == 'trainLabeled' or phase == 'trainUnlabeled':
                                if phase == 'trainLabeled':
                                    fig = visualize(inputs_vis[0][0], labels_vis[0], outputs_vis[0][0], figsize=(6, 6),
                                                    epoch=epoch, gray=False)
                                    plt.savefig(root_path + 'preview/train/Labeled/' + 'epoch_%s.jpg' % epoch)
                                elif phase == 'trainUnlabeled':
                                    fig = visualize(inputs_vis[0][0], labels_vis[0], outputs_vis[0][0], figsize=(6, 6),
                                                    epoch=epoch, gray=True)
                                    plt.savefig(root_path + 'preview/train/Unlabeled/' + 'epoch_%s.jpg' % epoch)
                                # plt.show(block=False)
                                # plt.pause(1.0)
                                plt.close(fig)

                    if phase == 'trainLabeled':
                        loss.backward(retain_graph=True)
                        optimizer[0].step()
                    elif phase == 'trainUnlabeled' and unet == False:
                        loss.backward()
                        if num_optimizer == 'two':
                            optimizer[1].step()
                        elif num_optimizer == 'one':
                            optimizer[0].step()

                running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss
                # compute loss
                if phase == 'trainLabeled':
                    epoch_train_loss[0] += epoch_loss
                elif phase == 'trainUnlabeled':
                    epoch_train_loss[1] += epoch_loss

                # compute validation accuracy, update training and validation loss, and calculate DICE and MSE
                if epoch % 20 == 19:
                    if phase == 'val_labeled':
                        running_val_dice, epoch_val_loss[0] = eval_net_dice(model, criterion, phase, network_switch, modelDataLoader['val_labeled'],
                                                    preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path)
                        epoch_val_dice = running_val_dice
                    elif phase == 'val_unlabeled':
                        running_val_mse, epoch_val_loss[1] = eval_net_mse(model, criterion, phase, network_switch, modelDataLoader['val_unlabeled'],
                                                    preview=PREVIEW, gpu=True, visualize_batch=0, epoch=epoch, slice=18, root_path=root_path)
                        epoch_val_mse = running_val_mse

                # # display TQDM information
                tqiter.set_description('MSSL (TSL=%.4f, TUL=%.4f, VSL=%.4f, VUL=%.4f, vdice=%.4f, vmse=%.4f)'
                                       % (epoch_train_loss[0]/(i+1), epoch_train_loss[1]/(i+1), epoch_val_loss[0], epoch_val_loss[1],
                                          epoch_val_dice, epoch_val_mse))

                # save and visualize training information
                if phase == 'val_unlabeled':
                    if epoch == 0:
                        title = 'Epoch   Train_L_loss   Train_U_loss   Val_L_loss   Val_U_loss   Val_dice   Val_MSE   ' \
                                'best_epoch\n'
                        cm.history_log(root_path + 'history_log.txt', title, 'w')
                        history = (
                            '{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.9f}     {:.4f}       {:d}\n'
                            .format(epoch, epoch_train_loss[0] / (i + 1), epoch_train_loss[1] / (i + 1), epoch_val_loss[0],
                                    epoch_val_loss[1], epoch_val_dice, epoch_val_mse, best_epoch))
                        cm.history_log(root_path + 'history_log.txt', history, 'a')

                        title = title.split()
                        data = history.split()
                        for ii, key in enumerate(title):
                            if ii == 0:
                                dict[key].append(int(data[ii]))
                            else:
                                dict[key].append(float(data[ii]))
                        visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
                        plt.savefig(root_path + 'Log.jpg')
                        plt.close(fig_loss)

                    else:
                        title = 'Epoch   Train_L_loss   Train_U_loss   Val_L_loss   Val_U_loss   Val_dice   Val_MSE   ' \
                                'best_epoch\n'
                        history = ('{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.9f}     {:.4f}       {:d}\n'
                               .format(epoch, epoch_train_loss[0]/(i+1), epoch_train_loss[1]/(i+1), epoch_val_loss[0], epoch_val_loss[1], epoch_val_dice, epoch_val_mse, best_epoch))
                        cm.history_log(root_path + 'history_log.txt', history, 'a')

                        title = title.split()
                        data = history.split()
                        for ii, key in enumerate(title):
                            if ii == 0:
                                dict[key].append(int(data[ii]))
                            else:
                                dict[key].append(float(data[ii]))
                        visualize_loss(fig_loss, dict=dict, title=title, epoch=epoch)
                        plt.savefig(root_path + 'Log.jpg')
                        plt.close(fig_loss)

                # save best validation model, figure preview and dice
                if phase == 'val_labeled' and (
                        epoch_val_dice > best_val_dice):
                        # or
                        # epoch_val_mse < best_val_mse):
                    best_epoch = epoch
                    best_val_dice = epoch_val_dice
                    # best_val_mse = epoch_val_mse
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), root_path + 'model/val_unet.pth')

                # if epoch % 100 == 99 and best_val_dice < 0.1:
                #     model.apply(ssl_3d_sep.weights_init)


    # compute run time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice: {:4f}'.format(best_val_dice))
    print('Best val MSE: {:4f}'.format(best_val_mse))
    model.load_state_dict(best_model_wts)
    return model, best_val_dice


# Set up training
def network_training_ssl_epoch(Test_only, job, data_seed, data_split, device, data_sizes, modelDataLoader,
                               num_epoch, folder_name, TSNE):

    val_dice = 0
    test_results = 0

    device = device
    dataset_sizes = data_sizes

    print('-' * 64)
    print('Training start')

    basic_path = folder_name + str(job) + '/' + str(data_split)

    if job == 'CNN_baseline':

        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': False,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': False}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)
        # model = ssl_3d_sep.semiSupervised3D_sep(1, 1, base_features).to(device)

        Pretrain = False

        # pretrain
        if Pretrain:
            model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))

        if not Test_only:

            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch, unet=True)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('CNN_baseline training finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('CNN_baseline training finished')

    elif job == 'MSSL_pretrain_Decoder':

        # # Autoencoder:
        switch = {'trainL_encoder': False,
                  'trainL_decoder_seg': False,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': True,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        if not Test_only:

            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch)

            # Save model and results
            torch.save(model.state_dict(), root_path + 'model/best_unet.pth')

            print('Autoencoder pretraining finished')

        # Decoder_Seg:
        switch = {'trainL_encoder': False,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': False,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': False}

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        # pretrain
        model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))

        # reset root path
        root_path = basic_path + '/seed' + str(data_seed) + '/' + 'Decoder_Seg/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        if not Test_only:

            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('Decoder_Seg training finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('Decoder_Seg testing finished')

    elif job == 'MSSL_pretrain_CNN':

        # Autoencoder:
        switch = {'trainL_encoder': False,
                  'trainL_decoder_seg': False,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': True,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch)

            # Save model and results
            torch.save(model.state_dict(), root_path + 'model/best_unet.pth')

            print('Autoencoder pretraining finished')

        # Decoder_Seg:
        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': False,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': False}

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        # pretrain
        model.load_state_dict(torch.load(root_path + 'model/best_unet.pth'))

        # reset root path
        root_path = basic_path + '/seed' + str(data_seed) + '/' + 'UNet/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('Pretrain UNet testing finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('Pretrain UNet testing finished')

    elif job == 'MSSL_alter_rec':
        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': False,

                  'trainU_encoder': True,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        Pretrain = False

        # pretrain
        if Pretrain:
            model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
            'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                job,
                data_seed,
                data_split,
                dataset_sizes,
                type(model),
                base_features,
                switch,
                criterion,
                optimizer_ft,
                exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                exp_lr_scheduler, num_epochs=num_epoch, num_optimizer='one')

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MSSL_alter_rec training finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MSSL_alter_rec training finished')

    elif job == 'MSSL_jointly':
        switch = {'trainL_encoder': True,
                  'trainL_decoder_seg': True,
                  'trainL_decoder_rec': True,

                  'trainU_encoder': True,
                  'trainU_decoder_seg': False,
                  'trainU_decoder_rec': True}

        root_path = basic_path + '/seed' + str(data_seed) + '/'
        cm.mkdir(root_path + 'model')
        cm.mkdir(root_path + 'preview')
        cm.mkdir(root_path + 'preview/train/Labeled')
        cm.mkdir(root_path + 'preview/train/Unlabeled')

        base_features = 16

        model = ssl_3d_sep.MSSL_norm(1, 1, base_features).to(device)

        Pretrain = False

        # pretrain
        if Pretrain:
            model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            # optimizer_ft = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0000)
            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
                'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\nbase features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
                'optimizer: {}\n\nlr scheduler: {}\n\n'.format(
                    job,
                    data_seed,
                    data_split,
                    dataset_sizes,
                    type(model),
                    base_features,
                    switch,
                    criterion,
                    optimizer_ft,
                    exp_lr_scheduler))

            cm.history_log(root_path + 'info.txt', train_info, 'w')

            print('data random seed: ', data_seed)
            print('device: ', device)
            print('dataset sizes: ', dataset_sizes)
            print('-' * 64)

            model, val_dice = train_model(model, modelDataLoader, device, root_path, switch, criterion, optimizer_ft,
                                          exp_lr_scheduler, num_epochs=num_epoch, jointly=True, self=False)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MSSL jointly training finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MSSL jointly testing finished')

    return val_dice, test_results





