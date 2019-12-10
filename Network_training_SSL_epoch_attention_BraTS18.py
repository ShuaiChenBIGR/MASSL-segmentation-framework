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
from module.dice_loss import DiceCoefficientLF
from module.visualize_attention import visualize_Seg, visualize_Rec, visualize_loss
from module.eval_attention_BraTS_slidingwindow import eval_net_dice, eval_net_mse, test_net_dice

from collections import defaultdict
from network import ssl_3d_attention
import time
import copy
from tqdm import trange

import warnings
warnings.filterwarnings('ignore')


def train_model(model, modelDataLoader, device, root_path, network_switch, criterion, optimizer, scheduler,
                num_epochs=25, loss_weighted=True, jointly=False, self=False, mode='fuse'):

    since = time.time()
    inputs = 0
    labels = 0
    image = 0
    image2 = 0
    outputsL = 0
    outputsU_back = 0
    outputsU_fore = 0
    labels_back = 0
    labels_fore = 0

    loss = 0

    PREVIEW = True

    dict = defaultdict(list)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_dice = 0.0
    best_val_mse = 1.0
    best_epoch = 0

    epoch_val_loss = np.array([0.0, 1.0])

    epoch_val_dice = 0.0
    epoch_val_mse = 1.0

    # set TQDM iterator
    tqiter = trange(num_epochs, desc='BraTS')

    for epoch in tqiter:

        epoch_train_loss = np.array([0.0, 1.0])
        fig_loss = plt.figure(num='loss', figsize=[12, 3.8])

        # go through all batches
        for i, (sample1, sample2) in enumerate(zip(modelDataLoader['trainLabeled'], modelDataLoader['trainUnlabeled'])):

            if i < (len(modelDataLoader['trainLabeled']) - 1) and i < (len(modelDataLoader['trainUnlabeled']) - 1):
                procedure = ['trainLabeled', 'trainUnlabeled']
            else:
                procedure = ['trainLabeled', 'trainUnlabeled', 'val_labeled', 'val_unlabeled']

            # run training and validation alternatively
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
                    inputs = sample1['image'][:, 2:3].float().to(device)  # batch, FLAIR
                    labels = sample1['mask'][:].float().to(device)
                    image = sample1['image'][:, 2:3].float().to(device)

                    if not self:
                        image2 = sample2['image'][:, 2:3].float().to(device)

                elif phase == 'trainUnlabeled':
                    inputs = sample2['image'][:, 2:3].float().to(device)
                    labels = sample2['mask'][:].float().to(device)   # batch, FLAIR
                    image = sample2['image'][:, 2:3].float().to(device)   # batch, FLAIR

                optimizer[0].zero_grad()
                optimizer[1].zero_grad()

                # update model parameters and compute loss
                with torch.set_grad_enabled(phase == 'trainLabeled' or phase == 'trainUnlabeled'):

                    if phase == 'trainLabeled':

                        outputsL, outputsU = model(inputs, phase=phase, network_switch=network_switch)

                        if mode == 'fuse':
                            outputsU_back = outputsU[:, 0]
                            outputsU_fore = outputsU[:, 1]
                            labels_back = (1.0 - outputsL) * image.float()
                            labels_fore = outputsL * image.float()
                            if not self:
                                labels_back = (1.0 - outputsL) * image2.float()
                                labels_fore = outputsL * image2.float()
                        elif mode == 'rec':
                            outputsU_back = outputsU[:, 0]
                            outputsU_fore = outputsU[:, 1]
                            labels_back = 1.0 * image.float()
                            labels_fore = 0.0 * image.float()
                            if not self:
                                labels_back = 1.0 * image2.float()
                                labels_fore = 0.0 * image2.float()
                        elif mode == 'seg':
                            outputsU_back = outputsU[:, 0]
                            outputsU_fore = outputsU[:, 1]
                            labels_back = 1.0 - outputsL
                            labels_fore = outputsL
                            if not self:
                                labels_back = 1.0 - outputsL
                                labels_fore = outputsL

                        w1 = 1.0
                        w2 = 0.0
                        w3 = 0.0

                        loss = w1 * criterion[0](outputsL.float(), labels.float()) + \
                            w2 * criterion[1](outputsU_back.float(), labels_back.float()) + \
                            w3 * criterion[1](outputsU_fore.float(), labels_fore.float())

                    elif phase == 'trainUnlabeled':

                        outputsL, outputsU = model(inputs, phase=phase, network_switch=network_switch)

                        if mode == 'fuse':
                            labels_back = (1.0 - outputsL) * image.float()
                            labels_fore = outputsL * image.float()
                        elif mode == 'rec':
                            labels_back = 1.0 * image.float()
                            labels_fore = 0.0 * image.float()
                        elif mode == 'seg':
                            labels_back = 1.0 - outputsL
                            labels_fore = outputsL

                        if loss_weighted:
                            w2 = torch.sum((1.0 - outputsL))
                            w3 = torch.sum(outputsL)
                            total = w2 + w3
                            w2 = w2 / total
                            w3 = w3 / total

                        # loss = criterion[1](outputsU.float(), outputsL.float())

                        loss = w2 * criterion[1](outputsU_back.float(), labels_back.float()) + \
                               w3 * criterion[1](outputsU_fore.float(), labels_fore.float())

                    outputsL_vis = outputsL.cpu().detach().numpy()
                    outputsU_back_vis = outputsU_back.cpu().detach().numpy()
                    outputsU_fore_vis = outputsU_fore.cpu().detach().numpy()
                    inputs_vis = inputs.cpu().detach().numpy()
                    labels_vis = labels.cpu().detach().numpy()
                    labels_back_vis = labels_back.cpu().detach().numpy()
                    labels_fore_vis = labels_fore.cpu().detach().numpy()

                    # visualize training set at the end of each epoch
                    if PREVIEW:
                        if i == (len(modelDataLoader['trainLabeled']) - 1):
                            if phase == 'trainLabeled' or phase == 'trainUnlabeled':

                                if phase == 'trainLabeled':
                                    fig = visualize_Seg(inputs_vis[0][0], labels_vis[0], outputsL_vis[0][0],
                                                        figsize=(6, 6), epoch=epoch)
                                    plt.savefig(root_path + 'preview/train/Labeled/' + 'epoch_%s.jpg' % epoch)
                                    plt.close(fig)

                                elif phase == 'trainUnlabeled':
                                    fig = visualize_Rec(inputs_vis[0][0], labels_back_vis[0, 0], labels_fore_vis[0, 0],
                                                        outputsU_back_vis[0], outputsU_fore_vis[0], figsize=(6, 6),
                                                        epoch=epoch)
                                    plt.savefig(root_path + 'preview/train/Unlabeled/' + 'epoch_%s.jpg' % epoch)
                                    plt.close(fig)

                    if phase == 'trainLabeled':
                        loss.backward(retain_graph=True)
                        optimizer[0].step()
                        running_loss += loss.item() * inputs.size(0)

                    elif phase == 'trainUnlabeled':
                        loss.backward()
                        optimizer[1].step()
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
                tqiter.set_description('MASSL (TSL=%.4f, TUL=%.4f, VSL=%.4f, VUL=%.4f, vdice=%.4f, vmse=%.4f)'
                                       % (epoch_train_loss[0]/(i+1), epoch_train_loss[1]/(i+1), epoch_val_loss[0], epoch_val_loss[1],
                                          epoch_val_dice, epoch_val_mse))

                # save and visualize training information
                if phase == 'val_unlabeled':
                    if epoch == 0:
                        title = 'Epoch   Train_L_loss   Train_U_loss   Val_L_loss   Val_U_loss   Val_dice   Val_MSE   ' \
                                'best_epoch\n'
                        cm.history_log(root_path + 'history_log.txt', title, 'w')
                        history = (
                            '{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.4f}     {:.4f}       {:d}\n'
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
                        history = ('{:3d}        {:.4f}         {:.4f}        {:.4f}       {:.4f}      {:.4f}     {:.4f}       {:d}\n'
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
                if phase == 'val_labeled' and (epoch_val_dice > best_val_dice):
                    best_epoch = epoch
                    best_val_dice = epoch_val_dice
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), root_path + 'model/val_unet.pth')

                if epoch % 200 == 199 and best_val_dice < 0.1:
                    model.apply(ssl_3d_attention.weights_init)

    # compute run time
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice: {:4f}'.format(best_val_dice))
    print('Best val MSE: {:4f}'.format(best_val_mse))
    model.load_state_dict(best_model_wts)
    return model, best_val_dice


# Set up training
def network_training_ssl_epoch(Test_only, job, data_seed, data_split, device, data_sizes, modelDataLoader, num_epoch, folder_name, TSNE):

    val_dice = 0
    test_results = 0

    device = device
    dataset_sizes = data_sizes

    print('-' * 64)
    print('Training start')

    basic_path = folder_name + str(job) + '/' + str(data_split)[:]

    #################################################
    if job == 'MASSL_alter':

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

        model = ssl_3d_attention.MASSL_norm(1, 1, base_features).to(device)

        use_existing = False

        if use_existing:
            model.load_state_dict(torch.load(root_path + 'model/val_unet.pth'))

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
            'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\n'
            'base features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
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
                                exp_lr_scheduler, num_epochs=num_epoch, loss_weighted=True)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MASSL_alter finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MASSL_alter finished')

    ###############################################

    elif job == 'MASSL_joint':

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

        model = ssl_3d_attention.MASSL_norm(1, 1, base_features).to(device)

        if not Test_only:
            criterionDICE = DiceCoefficientLF(device)
            criterionMSE = nn.MSELoss()
            criterion = (criterionDICE, criterionMSE)

            optimizer_ft = (optim.Adam(model.parameters(), lr=1e-2),
                            optim.Adam(model.parameters(), lr=1e-3))

            exp_lr_scheduler = (lr_scheduler.StepLR(optimizer_ft[0], step_size=500, gamma=0.5),
                                lr_scheduler.StepLR(optimizer_ft[1], step_size=500, gamma=0.5))

            # save training information
            train_info = (
                'job: {}\n\ndata random seed: {}\n\ndata_split: {}\n\ndataset sizes: {}\n\nmodel: {}\n\n'
                'base features: {}\n\nnetwork_switch: {}\n\nloss function: {}\n\n'
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
                                          exp_lr_scheduler, num_epochs=num_epoch, loss_weighted=True, jointly=True)

            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MASSL_outside_loss_jointly finished')

        else:
            # Testing model
            test_results = test_net_dice(root_path, basic_path, model, switch, modelDataLoader['test'], TSNE, gpu=True)
            print('MASSL_outside_loss_jointly finished')

    return val_dice, test_results






